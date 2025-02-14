# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import logging
import os
from collections import defaultdict
from datetime import datetime

import torch
from lm_eval import simple_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from bytelatent.args import EvalArgs, ValidationArgs
from bytelatent.checkpoint import CONSOLIDATE_FOLDER, consolidate_checkpoints
from bytelatent.config_parser import parse_args_to_pydantic_model
from bytelatent.data.file_util import get_fs
from bytelatent.distributed import (
    DistributedArgs,
    dist_mean_dict,
    get_global_rank,
    get_world_size,
    setup_torch_distributed,
)
from bytelatent.generate import (
    PackedCausalTransformerGenerator,
    load_consolidated_model_and_tokenizer,
)

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


def eval_on_val(generator, val_args: ValidationArgs, train_cfg):
    srcs = {}
    for src in val_args.sources:
        path = os.path.join(val_args.root_dir, src)
        srcs[path] = 1.0
    for src in train_cfg.data.sources:
        path = os.path.join(train_cfg.data.root_dir, src)
        srcs[path] = 1.0

    multi_state = init_choice_state(
        "", srcs, 0, get_global_rank(), get_world_size(), "*.val.jsonl"
    )
    path_to_iter = setup_sources(multi_state)

    max_gen_len = generator.max_gen_len
    # We temporarily lower max gen len
    generator.max_gen_len = 1

    all_val_metrics = {}
    for src in path_to_iter:
        jsonl_iterator = path_to_iter[src]
        texts = []
        logger.info(f"Running validation on {src}...")
        for step, (content, state) in enumerate(jsonl_iterator):
            if state["current_iter"] > 0 or (
                val_args.max_steps is not None and step >= val_args.max_steps
            ):
                break
            content_key = "text" if ("text" in content) else "content"
            texts.append(content[content_key])

        _, loglikelihood, _ = generator.generate(texts)

        metrics = defaultdict(list)
        for i, ll in enumerate(loglikelihood):
            tmp = ll.sum().item()
            metrics["nll"].append(tmp)
            metrics["nll_per_token"].append(tmp / len(ll))
            metrics["nll_per_char"].append(tmp / len(texts[i]))

            metrics["avg_seqlen"].append(len(ll))

        for m in metrics:
            metrics[m] = sum(metrics[m]) / len(metrics[m])
        metrics.update(dist_mean_dict(metrics))
        logger.info(f"Validation on {src} done. Metrics: {metrics}")

        name = os.path.basename(src)
        if name in all_val_metrics:
            logger.warning(
                f"Duplicate source name {name}, path {src} in validation sources, renaming to {name}_1"
            )
            name = f"{name}_1"
        all_val_metrics[name] = metrics

    generator.max_gen_len = max_gen_len

    return all_val_metrics


def launch_eval(eval_args: EvalArgs):
    if not torch.distributed.is_initialized():
        setup_torch_distributed(DistributedArgs())

    fs = get_fs(eval_args.ckpt_dir, s3_profile=eval_args.s3_profile)
    if (
        fs.exists(eval_args.ckpt_dir)
        and fs.exists(os.path.join(eval_args.ckpt_dir, "params.json"))
        and len(fs.glob(os.path.join(eval_args.ckpt_dir, "*.pth"))) != 0
    ):
        consolidate_path = eval_args.ckpt_dir
    else:
        consolidate_path = os.path.join(eval_args.ckpt_dir, CONSOLIDATE_FOLDER)
        if not fs.exists(consolidate_path) and get_global_rank() == 0:
            consolidate_path = consolidate_checkpoints(eval_args.ckpt_dir)

    fs.mkdirs(eval_args.dump_dir, exist_ok=True)
    with fs.open(os.path.join(eval_args.dump_dir, "config.yaml"), "w") as f:
        f.write(eval_args.model_dump_json())

    torch.distributed.barrier()
    logger.info("Loading model")
    # TODO: Make this general so that it works with either
    # LMTransformer or Blt, similar with args
    model, tokenizer, train_cfg = load_consolidated_model_and_tokenizer(
        consolidate_path,
    )
    logger.info("Model loaded")
    model.eval()
    generator = PackedCausalTransformerGenerator(eval_args.generator, model, tokenizer)

    wrap = EvalHarnessLM(generator)
    # Redo
    results = simple_evaluate(wrap, eval_args.harness.model_dump())
    val_results = None
    if eval_args.validation:
        val_results = eval_on_val(generator, eval_args.validation, train_cfg)
    if get_global_rank() == 0:
        with fs.open(os.path.join(eval_args.dump_dir, "results.json"), "w") as f:
            f.write(json.dumps(results))
        logger.info(f"All evaluation results: {results['results']}")
        if val_results is not None:
            with fs.open(os.path.join(eval_args.dump_dir, "validation.json"), "w") as f:
                f.write(json.dumps(val_results))
            logger.info(f"All validation results: {val_results}")
    if eval_args.metric_log_dir and get_global_rank() == 0:
        metric_log_path = os.path.join(eval_args.metric_log_dir, "metrics.eval.jsonl")

        logger.info(f"Writing metric logs to {metric_log_path}")
        timestamp = {
            "created_at": datetime.utcnow().isoformat(),
        }
        if eval_args.global_step is not None:
            timestamp["global_step"] = eval_args.global_step
        print(
            json.dumps(timestamp | results["results"]),
            file=fs.open(metric_log_path, mode="a"),
            flush=True,
        )

        val_log_path = os.path.join(
            eval_args.metric_log_dir, "metrics.validation.jsonl"
        )
        if val_results is not None:
            print(
                json.dumps(timestamp | val_results),
                file=fs.open(val_log_path, mode="a"),
                flush=True,
            )

    del generator


def main():
    eval_args = parse_args(EvalArgs)
    launch_eval(eval_args)


if __name__ == "__main__":
    main()
