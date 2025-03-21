# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import logging
import math
import os
from collections import defaultdict
from datetime import datetime

import torch
from lm_eval import simple_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from rich.progress import track
from torch.nn import functional as F

from bytelatent.args import (
    EvalArgs,
    TrainArgs,
    ValidationArgs,
    find_and_sanitize_chunks,
)
from bytelatent.checkpoint import CONSOLIDATE_FOLDER, consolidate_checkpoints
from bytelatent.config_parser import parse_args_to_pydantic_model
from bytelatent.data.file_util import get_fs
from bytelatent.data.iterators.arrow_iterator import ArrowFileIterator
from bytelatent.data.iterators.limit_iterator import LimitIterator
from bytelatent.data.iterators.packing_iterator import (
    PackingArgs,
    PackingIterator,
    PackingMode,
)
from bytelatent.data.iterators.preprocess_iterator import PreprocessIterator
from bytelatent.data.iterators.sequence_iterator import (
    SequenceIterator,
    SequencePackingArgs,
)
from bytelatent.data.patcher import PatcherArgs, PatchingModeEnum
from bytelatent.distributed import (
    DistributedArgs,
    dist_mean_dict,
    dist_sum,
    get_device_mesh,
    get_global_rank,
    get_world_size,
    setup_torch_distributed,
    to_py_num,
)
from bytelatent.generate import (
    PackedCausalTransformerGenerator,
    load_consolidated_model_and_tokenizer,
)
from bytelatent.model.blt import ByteLatentTransformer
from bytelatent.tokenizers.build_tokenizer import TokenizerArgs
from bytelatent.transformer import LMTransformer

EVAL_FOLDER_NAME = "{:010d}"

logger = logging.getLogger()


def all_dicts_same(dict_list):
    if not dict_list:  # Check if the list is empty
        return True

    # Compare each dictionary to the first one
    first_dict = dict_list[0]
    return all(d == first_dict for d in dict_list)


class MockAccelerator:
    def gather(self, tensor):
        l = [torch.zeros_like(tensor) for _ in range(get_world_size())]
        torch.distributed.all_gather(l, tensor)
        return torch.stack(l)

    def wait_for_everyone(self):
        torch.distributed.barrier()


# Light wrapper around generator for lm-eval harness
class EvalHarnessLM(LM):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator
        self.accelerator = MockAccelerator()
        self._rank = get_global_rank()
        self._world_size = get_world_size()
        self.device = generator.device

    def generate_until(self, requests: list[Instance]) -> list[str]:
        prompts, gen_args = zip(*[req.args for req in requests])
        assert all_dicts_same(gen_args), "Doesn't support different gen args for now"
        gen_args = gen_args[0]
        temperature = gen_args.get("temperature", 0.0)
        top_p = gen_args.get("top_p", None)
        top_k = gen_args.get("top_k", None)
        until = gen_args.get("until", [])

        self.generator.temperature = temperature
        self.generator.top_p = top_p
        self.generator.top_k = top_k
        self.generator.until = until
        generations, _, _ = self.generator.generate(prompts)
        filtered_gen = []
        for g in generations:
            for e in until:
                g = g.replace(e, "")
            filtered_gen.append(g)
        return filtered_gen

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        prompts, continuations = zip(*[req.args for req in requests])
        inputs = [req.args[0] + req.args[1] for req in requests]
        max_gen_len = self.generator.max_gen_len
        # We temporarily lower max gen len
        self.generator.max_gen_len = 1
        _, lls, greedy = self.generator.generate(inputs)
        results = []
        for p, ll, gr in zip(prompts, lls, greedy):
            p_len = len(
                self.generator.tokenizer.encode(p, add_bos=False, add_eos=False)
            )
            results.append((ll[p_len:].sum().item(), gr[p_len:].all().item()))

        self.generator.max_gen_len = max_gen_len
        return results

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        prompts = [req.args[0] for req in requests]
        max_gen_len = self.generator.max_gen_len
        # We temporarily lower max gen len
        self.generator.max_gen_len = 1
        _, lls, _ = self.generator.generate(prompts)
        results = []
        for ll in lls:
            results.append((ll.sum().item(),))
        self.generator.max_gen_len = max_gen_len

        return results


@torch.no_grad()
def eval_ppl_on_path(
    *,
    world_rank: int,
    world_size: int,
    model: LMTransformer | ByteLatentTransformer,
    tokenizer_args: TokenizerArgs,
    patcher_args: PatcherArgs,
    packing_args: PackingArgs,
    add_patches: bool,
    path: str,
    arrow_batch_size: int,
    max_n_docs: int | None,
    s3_profile: str | None = None,
):
    model.eval()
    seq_len = model.get_output_seq_len()
    arrow_iterator = ArrowFileIterator(
        file_path=None,
        dataset_files=[path],
        entropy_model_name=None,
        worker_id=world_rank,
        num_workers=world_size,
        arrow_batch_size=arrow_batch_size,
        preprocess_dir=None,
        s3_profile=s3_profile,
        file_format="arrow" if path.endswith("arrow") else "json",
    )
    if max_n_docs is not None:
        arrow_iterator = LimitIterator(arrow_iterator, limit=max_n_docs)
    preprocess_iterator = PreprocessIterator(
        arrow_iterator,
        patcher_args=patcher_args,
        tokenizer_args=tokenizer_args,
        add_patches=add_patches,
    )
    sequence_iterator = SequenceIterator(
        preprocess_iterator,
        sequence_packing_args=SequencePackingArgs(
            output_seq_len=seq_len,
            # Effectively disables shuffles
            buffer_size=1,
        ),
        rng_state=None,
    )
    packing_iterator = PackingIterator(sequence_iterator, packing_args=packing_args)
    total_loss = 0.0
    n_bytes = 0
    batch_iterator = packing_iterator.create_iter()
    for batch in batch_iterator:
        x = torch.from_numpy(batch.x).cuda()
        y = torch.from_numpy(batch.y).cuda()
        mask = None if batch.mask is None else torch.from_numpy(batch.mask).cuda()
        patch_lengths = batch.patch_lengths
        if patch_lengths is not None:
            patch_lengths = torch.from_numpy(patch_lengths).cuda()

        if tokenizer_args.name in ["bytes", "blt"]:
            n_bytes += y.numel() if mask is None else mask.sum().item()
            if isinstance(model, ByteLatentTransformer):
                pred = model(x, patch_lengths=patch_lengths)
            else:
                pred = model(x)
            loss = F.cross_entropy(pred.flatten(0, 1), y.flatten(0, 1), reduction="sum")
            total_loss += loss.item()
        else:
            raise NotImplementedError()
    all_n_bytes = to_py_num(dist_sum(n_bytes))
    all_total_loss = to_py_num(dist_sum(total_loss))
    return {
        "n_bytes": all_n_bytes,
        "n_bytes_gpu": n_bytes,
        "loss_sum": all_total_loss,
        "loss_sum_gpu": total_loss,
        "loss_mean": all_total_loss / all_n_bytes,
        "loss_mean_gpu": total_loss / n_bytes,
        "ppl": math.exp(all_total_loss / all_n_bytes) if all_n_bytes > 0 else 0.0,
        "bpb": all_total_loss / math.log(2) / all_n_bytes,
    }


def launch_eval(eval_args: EvalArgs):
    assert eval_args.dump_dir is not None
    assert eval_args.ckpt_dir is not None
    distributed_args = DistributedArgs()
    distributed_args.configure_world()
    if not torch.distributed.is_initialized():
        setup_torch_distributed(distributed_args)

    world_mesh = get_device_mesh(distributed_args)
    dp_mesh = world_mesh["dp_replicate"]
    assert distributed_args.dp_shard == 1
    world_size = dp_mesh.size()
    world_rank = dp_mesh.get_local_rank()

    fs = get_fs(eval_args.ckpt_dir, s3_profile=eval_args.s3_profile)
    if (
        fs.exists(eval_args.ckpt_dir)
        and fs.exists(os.path.join(eval_args.ckpt_dir, "params.json"))
        and len(fs.glob(os.path.join(eval_args.ckpt_dir, "*.pth"))) != 0
    ):
        consolidate_path = eval_args.ckpt_dir
    else:
        if eval_args.consolidate_if_needed:
            logger.info(
                "Found a model checkpoint, but it has not been consolidated.... so consolidating the checkpoint"
            )
            consolidate_path = os.path.join(
                eval_args.ckpt_dir, eval_args.consolidate_folder
            )
            if not fs.exists(consolidate_path) and get_global_rank() == 0:
                consolidate_path = consolidate_checkpoints(fs, eval_args.ckpt_dir)
            logger.info("Model consolidated to: %s", consolidate_path)
        else:
            raise ValueError(
                "Did not find a consolidated checkpoint and consolidate_if_needed is False"
            )

    fs.mkdirs(eval_args.dump_dir, exist_ok=True)
    with fs.open(os.path.join(eval_args.dump_dir, "config.yaml"), "w") as f:
        f.write(eval_args.model_dump_json())

    torch.distributed.barrier()
    logger.info("Loading model")
    model, tokenizer, train_cfg = load_consolidated_model_and_tokenizer(
        consolidate_path,
    )
    pad_id = 0 if train_cfg.data.tokenizer_args.name == "bytes" else tokenizer.boe_id
    model.eval()
    logger.info("Model loaded")

    ppl_results = None
    if eval_args.run_ppl:
        assert eval_args.validation is not None
        packing_args = PackingArgs(
            batch_size=eval_args.validation.batch_size,
            seq_len=train_cfg.data.seq_len,
            max_length=train_cfg.data.max_encoder_seq_length,
            pad_to_max_length=True,
            enable_byte_ngrams=False,
            pad_id=pad_id,
            packing_mode=(
                PackingMode.BYTES
                if train_cfg.data.patcher_args.patching_mode == PatchingModeEnum.byte
                else PackingMode.PATCHING
            ),
        )
        if len(eval_args.validation.sources) > 0:
            ppl_results = {}
            logger.info("Starting PPL evaluation on validation sets")
            for source in eval_args.validation.sources:
                ppl_results[source] = eval_ppl_on_path(
                    world_rank=world_rank,
                    world_size=world_size,
                    model=model,
                    tokenizer_args=train_cfg.data.tokenizer_args,
                    patcher_args=train_cfg.data.patcher_args,
                    packing_args=packing_args,
                    add_patches=train_cfg.data.add_patches,
                    path=os.path.join(eval_args.validation.root_dir, source),
                    max_n_docs=eval_args.validation.max_n_docs,
                    arrow_batch_size=20,
                    s3_profile=eval_args.s3_profile,
                )

    task_results = None
    if eval_args.run_tasks:
        assert eval_args.generator is not None
        assert eval_args.harness is not None
        generator = PackedCausalTransformerGenerator(
            eval_args.generator, model, tokenizer
        )
        wrap = EvalHarnessLM(generator)
        # TODO: This needs to be checked/sped up
        task_results = simple_evaluate(wrap, **eval_args.harness.model_dump())

    results = {"ppl": ppl_results, "tasks": task_results}
    # TODO: Serial and Parallel yield slightly different number of bytes, debug this later,
    # leaving this log statement here to help with that.
    # logging.info("Rank: %s Results: %s", world_rank, results)

    if get_global_rank() == 0:
        with fs.open(os.path.join(eval_args.dump_dir, "results.json"), "w") as f:
            f.write(json.dumps(results))
        logger.info(f"All evaluation results: {results}")
        if ppl_results is not None:
            with fs.open(os.path.join(eval_args.dump_dir, "validation.json"), "w") as f:
                f.write(json.dumps(ppl_results))
            logger.info(f"All validation results: {ppl_results}")

    if eval_args.metric_log_dir and get_global_rank() == 0:
        metric_log_path = os.path.join(eval_args.metric_log_dir, "metrics.eval.jsonl")

        logger.info(f"Writing metric logs to {metric_log_path}")
        timestamp: dict[str, int | str] = {
            "created_at": datetime.utcnow().isoformat(),
        }
        if eval_args.global_step is not None:
            timestamp["global_step"] = eval_args.global_step
        print(
            json.dumps(timestamp | results),
            file=fs.open(metric_log_path, mode="a"),
            flush=True,
        )

        val_log_path = os.path.join(
            eval_args.metric_log_dir, "metrics.validation.jsonl"
        )
        if ppl_results is not None:
            print(
                json.dumps(timestamp | ppl_results),
                file=fs.open(val_log_path, mode="a"),
                flush=True,
            )


def main():
    eval_args = parse_args_to_pydantic_model(EvalArgs)
    launch_eval(eval_args)


if __name__ == "__main__":
    main()
