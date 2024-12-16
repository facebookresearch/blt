from setuptools import setup, find_packages

setup(
    name="bytelatent",
    version="0.1.0",
    description="Byte Latent Transformer: Patches Scale Better Than Tokens",
    author="Meta Platforms, Inc. and affiliates.",
    url="https://github.com/facebookresearch/blt",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "omegaconf",
        "msgspec",
        "rouge-score",
        "sacrebleu",
        "sentencepiece",
        "tiktoken",
        "fsspec",
        "blobfile",
        "wandb",
        "viztracer",
        "lm-eval",
        "scipy",
        "pynvml",
        "datatrove",
        "orjson",
        "luigi",
        "pydantic",
        "altair",
        "submitit",
        "typer",
        "rich",
        "xformers"
    ]
)