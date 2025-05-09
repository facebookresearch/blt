# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import os
import shutil
import subprocess
from typing import Any, Dict

import jinja2
from omegaconf import OmegaConf
from pydantic import BaseModel

from bytelatent.config_parser import parse_args_to_pydantic_model


class StoolArgs(BaseModel):
    name: str
    dump_dir: str
    # model_config is a reserved name by pydantic, so use this instead
    model_conf: Any = None
    launcher: str = "sbatch"  # Can be sbatch or bash if already in salloc
    python_command: str = "python"
    use_conda: bool = True
    script: str = "apps.main.train"  # The script to run.
    copy_code: bool = True  # Wether to copy code to dump dir
    dirs_exists_ok: bool = (
        False  # Wether to copy new code and config and run regardless that dir exists
    )
    override: bool = False  # Whether to delete dump dir and restart, requires confirmation
    force_override: bool = False # Does not require interaction
    nodes: int = -1  # The number of nodes to run the job on.
    ngpu: int = 8  # The number of GPUs required per node.
    ncpu: int = 16  # The number of CPUs allocated per GPU.
    mem: str = ""  # The amount of memory to allocate.
    anaconda: str = "default"  # The path to the anaconda environment.
    constraint: str = ""  # The constraint on the nodes.
    exclude: str = ""  # The nodes to exclude.
    time: int = -1  # The time limit of the job (in minutes).
    account: str = ""
    qos: str = ""
    partition: str = "learn"
    stdout: bool = False
    dry_run: bool = False



def copy_dir(input_dir: str, output_dir: str) -> None:
    print(f"Copying : {input_dir}\n" f"to      : {output_dir} ...")
    assert os.path.isdir(input_dir), f"{input_dir} is not a directory"
    assert os.path.isdir(output_dir), f"{output_dir} is not a directory"
    rsync_cmd = (
        f"rsync -rmt --copy-links "
        f"--exclude .venv "
        f"--include '**/' "
        f"--include '*.py' "
        f"--exclude='*' "
        f"{input_dir}/ {output_dir}"
    )
    print(f"Copying command: {rsync_cmd}")
    subprocess.call([rsync_cmd], shell=True)
    print("Copy done.")


def retrieve_max_time_per_partition() -> Dict[str, int]:
    # retrieve partition max times (a bit slow)

    sinfo = json.loads(subprocess.check_output("sinfo --json", shell=True))["sinfo"]
    max_times: Dict[str, int] = {}

    for info in sinfo:
        if info["partition"]["maximums"]["time"]["infinite"]:
            max_times[info["partition"]["name"]] = 14 * 24 * 60  # 14 days
        else:
            max_times[info["partition"]["name"]] = info["partition"]["maximums"][
                "time"
            ][
                "number"
            ]  # in minutes

    return max_times


def validate_args(args) -> None:
    # Set maximum time limit if not specified
    if args.time == -1:
        max_times = retrieve_max_time_per_partition()
        args.time = max_times.get(
            args.partition, 3 * 24 * 60
        )  # Default to 3 days if not found
        print(
            f"No time limit specified, using max time for partitions: {args.time} minutes"
        )

    if args.constraint:
        args.constraint = f"#SBATCH --constraint={args.constraint}"

    if args.account:
        args.account = f"#SBATCH  --account={args.account}"

    if args.qos:
        args.qos = f"#SBATCH --qos={args.qos}"

    if getattr(args, "exclude", ""):
        args.exclude = f"#SBATCH --exclude={args.exclude}"

    if hasattr(args, "anaconda") and args.anaconda:
        if args.anaconda == "default":
            args.anaconda = (
                subprocess.check_output("which python", shell=True)
                .decode("ascii")
                .strip()
            )
        else:
            args.anaconda = f"{args.anaconda}/bin/python"
        assert os.path.isfile(args.anaconda)

    args.mem = args.mem or "0"

    assert args.partition
    assert args.ngpu > 0
    assert args.ncpu > 0
    assert args.nodes > 0
    assert args.time > 0
    assert args.partition


def launch_job(args: StoolArgs):
    # Set up args default and validate them depending on the cluster or partition requested
    validate_args(args)
    job_name = args.name or args.model_conf["name"]
    dump_dir = os.path.join(args.dump_dir, job_name) or args.model_conf["dump_dir"]
    print("Creating directories...")
    os.makedirs(dump_dir, exist_ok=args.dirs_exists_ok or args.override or args.force_override)
    if args.override or args.force_override:
        if args.force_override:
            shutil.rmtree(dump_dir)
            print(f"Directory '{dump_dir}' has been deleted.")
        else:
            confirm = input(
                f"Are you sure you want to delete the directory '{dump_dir}'? This action cannot be undone. (yes/no): "
            )
            if confirm.lower() == "yes":
                shutil.rmtree(dump_dir)
                print(f"Directory '{dump_dir}' has been deleted.")
            else:
                print("Operation cancelled.")
                return
    if args.copy_code:
        os.makedirs(f"{dump_dir}/code", exist_ok=args.dirs_exists_ok)
        print("Copying code ...")
        copy_dir(os.getcwd(), f"{dump_dir}/code")

    print("Saving config file ...")
    shutil.copy(args.model_conf, f"{dump_dir}/base_config.yaml")

    conda_exe = os.environ.get("CONDA_EXE", "conda")
    conda_env_path = os.path.dirname(os.path.dirname(args.anaconda))
    log_output = (
        "-o $DUMP_DIR/logs/%j/%j_%t.out -e $DUMP_DIR/logs/%j/%j_%t.err"
        if not args.stdout
        else ""
    )
    env = jinja2.Environment(
        loader=jinja2.PackageLoader('bytelatent'),
        autoescape=jinja2.select_autoescape(),
    )
    template = env.get_template('stool_template.sh.jinja')
    sbatch_jinja = template.render(
        name=job_name,
        script=args.script,
        dump_dir=dump_dir,
        nodes=args.nodes,
        tasks=args.nodes * args.ngpu,
        nodes_per_run=args.nodes,
        ngpus=args.ngpu,
        ncpu=args.ncpu,
        mem=args.mem,
        qos=args.qos,
        account=args.account,
        constraint=args.constraint,
        exclude=args.exclude,
        time=args.time,
        partition=args.partition,
        python_command=args.python_command,
        conda_exe=conda_exe,
        conda_env_path=conda_env_path,
        use_conda=args.use_conda,
        log_output=log_output,
        go_to_code_dir=f"cd {dump_dir}/code/" if args.copy_code else "",
    )

    print("Writing sbatch command ...")
    with open(f"{dump_dir}/submit.slurm", "w") as f:
        f.write(sbatch_jinja)

    if args.dry_run:
        print("Dry run mode enabled. Not submitting job.")
    else:
        print("Submitting job ...")
        os.system(f"{args.launcher} {dump_dir}/submit.slurm")

    print("Done.")


if __name__ == "__main__":
    """
    The command line interface here uses OmegaConf https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments
    """
    args = parse_args_to_pydantic_model(StoolArgs, instantiate_default_cls=False)
    launch_job(args)
