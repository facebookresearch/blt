import copy
from typing import Type, TypeVar

import omegaconf
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel


def parse_file_config(path: str) -> DictConfig:
    file_cfg = OmegaConf.load(path)
    if not isinstance(file_cfg, DictConfig):
        raise ValueError(
            f"File paths must parse to DictConfig, but it was: {type(file_cfg)}"
        )
    return file_cfg


def recursively_parse_config(cfg: DictConfig) -> list[DictConfig]:
    if "config" not in cfg:
        return [cfg]

    ordered_cfgs = []
    cfg = copy.deepcopy(cfg)
    config_arg = cfg["config"]
    del cfg["config"]
    ordered_cfgs.append(cfg)

    if isinstance(config_arg, str):
        file_cfg = parse_file_config(config_arg)
        sub_configs = recursively_parse_config(file_cfg)
        ordered_cfgs = sub_configs + ordered_cfgs
    elif isinstance(config_arg, omegaconf.listconfig.ListConfig):
        sub_configs = []
        for c in config_arg:
            if not isinstance(c, str):
                raise ValueError(
                    f'If "config" is specified, it must be either a string path or a list of string paths. It was config={config_arg}'
                )
            config_to_parse = parse_file_config(c)
            sub_configs.extend(recursively_parse_config(config_to_parse))
        ordered_cfgs = sub_configs + ordered_cfgs
    else:
        raise ValueError(
            f'If "config" is specified, it must be either a string path or a list of string paths, it was config={config_arg}'
        )
    return ordered_cfgs


def parse_args_with_default(
    *, default_cfg: DictConfig | None = None, cli_args: DictConfig | None = None
):
    if cli_args is None:
        cli_args = OmegaConf.from_cli()
        assert isinstance(
            cli_args, DictConfig
        ), f"CLI Args must be a DictConfig, not {type(cli_args)}"
    ordered_cfgs = recursively_parse_config(cli_args)
    if default_cfg is not None:
        ordered_cfgs.insert(0, default_cfg)
    cfg = OmegaConf.merge(*ordered_cfgs)
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


T = TypeVar("T", bound=BaseModel)


def parse_args_to_pydantic_model(
    args_cls: Type[T], cli_args: DictConfig | None = None
) -> T:
    default_cfg = OmegaConf.create(args_cls().model_dump())
    parsed_cfg = parse_args_with_default(default_cfg=default_cfg, cli_args=cli_args)
    pydantic_args = args_cls.model_validate(parsed_cfg)
    return pydantic_args
