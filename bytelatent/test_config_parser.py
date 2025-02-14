import os

import pytest
from omegaconf import DictConfig, MissingMandatoryValue, OmegaConf
from pydantic import BaseModel, ConfigDict

from bytelatent.config_parser import (
    parse_args_to_pydantic_model,
    parse_file_config,
    recursively_parse_config,
)

FIXTURE_DIR = "fixtures/test-cfgs"


def test_parse_file_config():
    with pytest.raises(ValueError):
        cfg = parse_file_config(os.path.join(FIXTURE_DIR, "list.yaml"))
        assert isinstance(cfg, DictConfig)


def test_nop():
    cfg = OmegaConf.create({"a": 1})
    parsed_cfgs = recursively_parse_config(cfg)
    assert len(parsed_cfgs) == 1
    assert parsed_cfgs[0] == cfg


def test_root():
    cli_cfg = OmegaConf.create({"config": os.path.join(FIXTURE_DIR, "root.yaml")})
    parsed_cfgs = recursively_parse_config(cli_cfg)
    assert len(parsed_cfgs) == 2
    assert len(parsed_cfgs[1]) == 0
    assert parsed_cfgs[0]["seed"] == -1
    with pytest.raises(MissingMandatoryValue):
        assert parsed_cfgs[0]["b"]["y"] is not None

    # Test basic cli override
    cli_cfg = OmegaConf.create(
        {"config": os.path.join(FIXTURE_DIR, "root.yaml"), "seed": 42}
    )
    parsed_cfgs = recursively_parse_config(cli_cfg)
    assert parsed_cfgs[1]["seed"] == 42
    cfg = OmegaConf.merge(*parsed_cfgs)
    assert cfg["seed"] == 42


def test_one_level_include():
    cli_cfg = OmegaConf.create({"config": os.path.join(FIXTURE_DIR, "middle.yaml")})
    parsed_cfgs = recursively_parse_config(cli_cfg)
    assert len(parsed_cfgs) == 3
    assert parsed_cfgs[0]["seed"] == -1
    assert parsed_cfgs[1]["b"]["y"] == 10
    assert len(parsed_cfgs[2]) == 0
    cfg = OmegaConf.merge(*parsed_cfgs)
    assert cfg["b"]["y"] == 10

    cli_cfg = OmegaConf.create(
        {"config": os.path.join(FIXTURE_DIR, "middle.yaml"), "b": {"y": 100}}
    )
    parsed_cfgs = recursively_parse_config(cli_cfg)
    assert len(parsed_cfgs) == 3
    assert parsed_cfgs[0]["seed"] == -1
    assert parsed_cfgs[1]["b"]["y"] == 10
    assert parsed_cfgs[2]["b"]["y"] == 100
    cfg = OmegaConf.merge(*parsed_cfgs)
    assert cfg["b"]["y"] == 100


def test_two_level_include():
    cli_cfg = OmegaConf.create(
        {"config": os.path.join(FIXTURE_DIR, "top.yaml"), "p": 500, "b": {"z": -2}}
    )
    parsed_cfgs = recursively_parse_config(cli_cfg)
    assert len(parsed_cfgs) == 4
    assert parsed_cfgs[0]["seed"] == -1
    assert parsed_cfgs[1]["b"]["y"] == 10
    assert parsed_cfgs[2]["hello"] == "world"
    assert parsed_cfgs[3]["p"] == 500
    assert parsed_cfgs[3]["b"]["z"] == -2
    cfg = OmegaConf.merge(*parsed_cfgs)
    assert cfg["a"] == 1
    assert cfg["seed"] == -1
    assert cfg["b"]["x"] == 0
    assert cfg["b"]["y"] == 10
    assert cfg["b"]["z"] == -2
    assert cfg["hello"] == "world"


def test_multiple_includes():
    cli_cfg = OmegaConf.create(
        {
            "config": [
                os.path.join(FIXTURE_DIR, "top.yaml"),
                os.path.join(FIXTURE_DIR, "override.yaml"),
            ],
            "p": 500,
            "b": {"z": -2},
        }
    )
    parsed_cfgs = recursively_parse_config(cli_cfg)
    assert len(parsed_cfgs) == 5
    assert parsed_cfgs[0]["seed"] == -1
    assert parsed_cfgs[1]["b"]["y"] == 10
    assert parsed_cfgs[2]["hello"] == "world"
    assert parsed_cfgs[3]["a"] == 100
    assert parsed_cfgs[4]["p"] == 500
    assert parsed_cfgs[4]["b"]["z"] == -2
    cfg = OmegaConf.merge(*parsed_cfgs)
    assert cfg["a"] == 100
    assert cfg["seed"] == -1
    assert cfg["b"]["x"] == 0
    assert cfg["b"]["y"] == 10
    assert cfg["b"]["z"] == -2
    assert cfg["hello"] == "world"

    cli_cfg = OmegaConf.create(
        {
            "config": [
                os.path.join(FIXTURE_DIR, "top.yaml"),
                os.path.join(FIXTURE_DIR, "override.yaml"),
            ],
            "p": 500,
            "b": {"z": -2},
            "a": 1000,
        }
    )
    parsed_cfgs = recursively_parse_config(cli_cfg)
    assert len(parsed_cfgs) == 5
    assert parsed_cfgs[0]["seed"] == -1
    assert parsed_cfgs[1]["b"]["y"] == 10
    assert parsed_cfgs[2]["hello"] == "world"
    assert parsed_cfgs[3]["a"] == 100
    assert parsed_cfgs[4]["p"] == 500
    assert parsed_cfgs[4]["b"]["z"] == -2
    cfg = OmegaConf.merge(*parsed_cfgs)
    assert cfg["a"] == 1000
    assert cfg["seed"] == -1
    assert cfg["b"]["x"] == 0
    assert cfg["b"]["y"] == 10
    assert cfg["b"]["z"] == -2
    assert cfg["hello"] == "world"


class SubConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: int = -100
    y: int = -100
    z: int = -5


class SampleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    a: int = -100
    seed: int = -100
    b: SubConfig = SubConfig()
    hello: str = ""
    p: int = -100


def test_pydantic_parse():
    cli_cfg = OmegaConf.create(
        {
            "config": [
                os.path.join(FIXTURE_DIR, "top.yaml"),
                os.path.join(FIXTURE_DIR, "override.yaml"),
            ],
            "p": 500,
            "a": 1000,
        }
    )
    cfg = parse_args_to_pydantic_model(SampleConfig, cli_args=cli_cfg)
    assert isinstance(cfg, SampleConfig)
    assert cfg.a == 1000
    assert cfg.p == 500
    assert cfg.seed == -1
    assert cfg.b.x == 0
    assert cfg.b.y == 10
    assert cfg.b.z == -5
    assert cfg.hello == "world"
