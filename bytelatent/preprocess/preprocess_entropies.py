# Copyright (c) Meta Platforms, Inc. and affiliates.
import time

import fsspec
import jsonlines
import numpy as np
import pyarrow as pa
import torch
import typer
from rich.progress import Progress, TextColumn

from bytelatent.data.file_util import get_fs
from bytelatent.data.patcher import calculate_entropies
from bytelatent.entropy_model import load_entropy_model
from bytelatent.tokenizers.build_tokenizer import TokenizerArgs


def get_id_from_doc(doc: dict) -> int:
    """
    We need a reliable way to ensure that samples from jsonl
    and arrow are the same, but there is no unique id field,
    so derive the best possible
    """
    if "sample_id" in doc:
        sample_id = doc["sample_id"]
    elif "title" in doc:
        sample_id = doc["title"]
    elif "qid" in doc:
        sample_id = doc["qid"]
    elif "paper_id" in doc:
        sample_id = doc["paper_id"]
    elif "path" in doc:
        sample_id = doc["path"]
    elif "url" in doc:
        sample_id = doc["url"]
    elif "id" in doc:
        sample_id = doc["id"]
    else:
        raise ValueError(f"Could not find a id key from: {doc.keys()}")
    return str(sample_id)


def get_text(doc: dict):
    if "text" in doc:
        text = doc["text"]
    elif "content" in doc:
        text = doc["content"]
    else:
        raise ValueError(f"Could not find a text key from: {doc.keys()}")
    return text


def jsonl_file_iterator(fs: fsspec.AbstractFileSystem, path: str):
    with fs.open(path) as f:
        reader = jsonlines.Reader(f)
        yield from reader


def main(
    input_file: str,
    output_file: str,
    patching_device: str = "cuda",
    log_step: int = 10_000,
    entropy_model_checkpoint_dir: str = "public_data/entropy_checkpoint",
    entropy_model_state_dict_path: str = "public_data/entropy_model.pth",
    bpe_tokenizer_path: str = "public_data/tokenizer.model",
    dry_run: bool = False,
    s3_profile: str | None = None,
):
    print(f"Preprocessing entropies, input: {input_file}, output: {output_file}")
    print("Loading entropy model", entropy_model_checkpoint_dir)
    input_fs = get_fs(input_file, s3_profile=s3_profile)
    input_doc_iterator = jsonl_file_iterator(input_fs, input_file)

    if dry_run:
        return
    entropy_model = load_entropy_model(
        entropy_model_checkpoint_dir,
        entropy_model_state_dict_path,
        device=patching_device,
    )

    print("Creating patcher")
    patching_batch_size = 32
    print("Creating tokenizer")
    tokenizer_args = TokenizerArgs(
        name="blt", init_kwargs={"bpe_tokenizer_path": bpe_tokenizer_path}
    )
    tokenizer = tokenizer_args.build()
    step = 0
    print("starting")
    start_time = time.time()
    patch_time = 0
    entropy_field = pa.field("entropies", pa.list_(pa.float16()), nullable=False)
    sample_id_field = pa.field("sample_id", pa.string(), nullable=False)
    text_field = pa.field("text", pa.string(), nullable=False)
    schema = pa.schema([sample_id_field, text_field, entropy_field])
    arrow_batch_size = 1_000

    output_fs = get_fs(output_file, s3_profile=s3_profile)

    try:
        with output_fs.open(output_file, "wb") as sink:
            with pa.ipc.new_file(sink, schema) as writer:
                id_buffer = []
                entropies_buffer = []
                text_buffer = []
                with Progress(
                    *Progress.get_default_columns(),
                    TextColumn("Completed: {task.completed}"),
                ) as progress:
                    task = progress.add_task(
                        "[green]Calculating entropies...", total=None
                    )
                    for doc in input_doc_iterator:
                        sample_id = get_id_from_doc(doc)
                        text = get_text(doc)
                        tokens = torch.tensor(tokenizer.encode(text))
                        patch_start = time.time()
                        scores, _ = calculate_entropies(
                            tokens,
                            entropy_model,
                            patching_batch_size,
                            patching_device,
                        )
                        entropies_buffer.append(
                            np.array(scores.tolist(), dtype=np.float16)
                        )
                        id_buffer.append(sample_id)
                        text_buffer.append(text)
                        if len(entropies_buffer) == arrow_batch_size:
                            batch = pa.record_batch(
                                {
                                    "entropies": entropies_buffer,
                                    "sample_id": id_buffer,
                                    "text": text_buffer,
                                },
                                schema,
                            )
                            writer.write(batch)
                            entropies_buffer = []
                            id_buffer = []
                            text_buffer = []
                        patch_time += time.time() - patch_start
                        step += 1
                        if step % log_step == 0:
                            print("Completed steps:", step)
                        progress.update(task, advance=1)
                    if len(entropies_buffer) > 0:
                        # Write last things
                        batch = pa.record_batch(
                            {
                                "entropies": entropies_buffer,
                                "sample_id": id_buffer,
                                "text": text_buffer,
                            },
                            schema,
                        )
                        writer.write(batch)
                        entropies_buffer = []
                        id_buffer = []
                        text_buffer = []
        output_fs.touch(f"{output_file}.complete")
    except:
        if output_fs.exists(output_file):
            output_fs.rm(output_file)
        raise
    elapsed = time.time() - start_time
    print("steps", step)
    print("done in:", elapsed)


if __name__ == "__main__":
    typer.run(main)
