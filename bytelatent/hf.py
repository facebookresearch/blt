import os
from typing import Union, Optional, Dict, Any
from pathlib import Path
import json
from dataclasses import asdict, is_dataclass

from huggingface_hub.hub_mixin import ModelHubMixin, DataclassInstance
from huggingface_hub import snapshot_download, constants
import typer
import torch

from bytelatent.distributed import DistributedArgs, setup_torch_distributed
from bytelatent.generate_blt import generate_nocache
from bytelatent.entropy_model import load_entropy_model
from bytelatent.model.blt import ByteLatentTransformer
from bytelatent.transformer import LMTransformer
from bytelatent.tokenizers.blt_tokenizer import BltTokenizer

from bytelatent.generate import load_consolidated_model_and_tokenizer


app = typer.Typer()


class BltModelWrapper(ModelHubMixin):
    def __init__(self, checkpoint_dir: str):
        self.model, self.tokenizer, self.train_cfg = (
            load_consolidated_model_and_tokenizer(checkpoint_dir)
        )
        assert isinstance(self.model, ByteLatentTransformer)
        assert isinstance(self.tokenizer, BltTokenizer)
        self.patcher_args = self.train_cfg.data.patcher_args.model_copy(deep=True)
        self.patcher_args.realtime_patching = True
        self.patcher_args.entropy_model_checkpoint_dir = os.path.join(
            checkpoint_dir, "entropy_model"
        )
        self.patcher = self.patcher_args.build()

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        force_download: bool = False,
        proxies: dict | None = None,
        resume_download: bool | None = None,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ):
        if os.path.isdir(model_id):
            model = cls(model_id)
        else:
            checkpoint_dir = snapshot_download(
                model_id,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
            )
            model = cls(checkpoint_dir)
        return model

    # Copied from https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/hub_mixin.py#L76
    # So that we can remove behavior we don't want, specifically:
    # - Overwriting the model card should not be allowed, any changes to the facebook/blt and related model cards should be done by hand and verified.
    # - Push to hub should be disabled, this also should be done by hand and verified.
    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        *,
        config: Optional[Union[dict, DataclassInstance]] = None,
        repo_id: Optional[str] = None,
        push_to_hub: bool = False,
        model_card_kwargs: Optional[Dict[str, Any]] = None,
        **push_to_hub_kwargs,
    ) -> Optional[str]:
        """
        Save weights in local directory.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the model weights and configuration will be saved.
            config (`dict` or `DataclassInstance`, *optional*):
                Model configuration specified as a key/value dictionary or a dataclass instance.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Huggingface Hub after saving it.
            repo_id (`str`, *optional*):
                ID of your repository on the Hub. Used only if `push_to_hub=True`. Will default to the folder name if
                not provided.
            model_card_kwargs (`Dict[str, Any]`, *optional*):
                Additional arguments passed to the model card template to customize the model card.
            push_to_hub_kwargs:
                Additional key word arguments passed along to the [`~ModelHubMixin.push_to_hub`] method.
        Returns:
            `str` or `None`: url of the commit on the Hub if `push_to_hub=True`, `None` otherwise.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Remove config.json if already exists. After `_save_pretrained` we don't want to overwrite config.json
        # as it might have been saved by the custom `_save_pretrained` already. However we do want to overwrite
        # an existing config.json if it was not saved by `_save_pretrained`.
        config_path = save_directory / constants.CONFIG_NAME
        config_path.unlink(missing_ok=True)

        # save model weights/files (framework-specific)
        self._save_pretrained(save_directory)

        # save config (if provided and if not serialized yet in `_save_pretrained`)
        if config is None:
            config = self._hub_mixin_config
        if config is not None:
            if is_dataclass(config):
                config = asdict(config)  # type: ignore[arg-type]
            if not config_path.exists():
                config_str = json.dumps(config, sort_keys=True, indent=2)
                config_path.write_text(config_str)

        return None

    def push_to_hub(self, *args, **kwargs):
        raise ValueError(
            "For meta authors: Do not push BLT weights with this, save weights with save_pretrained() then push them manually to HF hub to ensure the repository metadata is correct."
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        raise ValueError(
            "Not needed for loading pre-trained weights, but nice to have later"
        )


@app.command()
def convert_to_transformers(blt_weights_dir: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    model, tokenizer, train_cfg = load_consolidated_model_and_tokenizer(blt_weights_dir)
    blt_dir = os.path.join(output_dir, "blt")
    entropy_dir = os.path.join(output_dir, "entropy")
    model.save_pretrained(blt_dir)
    blt_readme_file = os.path.join(blt_dir, "README.md")
    if os.path.exists(blt_readme_file):
        os.remove(blt_readme_file)

    patcher_args = train_cfg.data.patcher_args.model_copy(deep=True)
    patcher_args.realtime_patching = False
    print("Loading entropy model and patcher")
    patcher_args.entropy_model_checkpoint_dir = os.path.join(
        blt_weights_dir, "entropy_model"
    )
    patcher = patcher_args.build()
    state_path = os.path.join(
        patcher_args.entropy_model_checkpoint_dir, "consolidated.pth"
    )
    entropy_model = load_entropy_model(
        patcher_args.entropy_model_checkpoint_dir, state_path
    )
    entropy_model.save_pretrained(entropy_dir)
    entropy_readme_file = os.path.join(entropy_dir, "README.md")
    if os.path.exists(entropy_readme_file):
        os.remove(entropy_readme_file)

    # TODO: Persist tokenizer in HF compatible way


@app.command()
def load_custom(
    blt_repo: str = "facebook/blt-1b",
):
    distributed_args = DistributedArgs()
    distributed_args.configure_world()
    if not torch.distributed.is_initialized():
        setup_torch_distributed(distributed_args)
    blt = BltModelWrapper.from_pretrained(blt_repo)
    prompts = ["The answer is"]
    outputs = generate_nocache(
        prompts, model=blt.model, tokenizer=blt.tokenizer, patcher=blt.patcher
    )
    text_outputs = [blt.tokenizer.decode(t) for t in outputs]
    for p, t in zip(prompts, text_outputs):
        print(f'Prompt: "{p}" Completion: "{t}"')
        print()


@app.command()
def load_transformers(
    source: str,
    entropy_repo: str = "facebook/blt-entropy",
    blt_repo: str = "facebook/blt-1b",
    entropy_dir: str | None = None,
    blt_dir: str | None = None,
):
    if source == "local":
        assert entropy_dir is not None
        assert blt_dir is not None
        entropy_model = LMTransformer.from_pretrained(
            entropy_dir, local_files_only=True
        )
        blt_model = ByteLatentTransformer.from_pretrained(
            blt_dir, local_files_only=True
        )
    elif source == "hub":
        entropy_model = LMTransformer.from_pretrained(entropy_repo)
        blt_model = ByteLatentTransformer.from_pretrained(blt_repo)
    else:
        raise ValueError(f"Unknown source: {source}")

    # TODO: Need a way to get tokenizer
    # TODO: Need a way to get patching settings
    # TODO: Insert test inference call


if __name__ == "__main__":
    app()
