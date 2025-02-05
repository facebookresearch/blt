import logging
import os
import shutil
import time
from enum import Enum

import fsspec
import submitit
import typer
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger("parallel_copy")


S3_PREFIX = "s3://"


def get_fs(path: str, s3_profile: str | None = None) -> fsspec.AbstractFileSystem:
    if path.startswith("s3://"):
        if s3_profile is None:
            return fsspec.filesystem(
                "s3", default_block_size=1000000 * 2**20, max_concurrency=1
            )
        else:
            return fsspec.filesystem(
                "s3",
                profile=s3_profile,
                default_block_size=1000000 * 2**20,
                max_concurrency=1,
            )
    else:
        return fsspec.filesystem("file")


def strip_s3_prefix(path: str):
    if path.startswith(S3_PREFIX):
        return path[len(S3_PREFIX) :]
    else:
        return path


class OverwriteMode(str, Enum):
    ALWAYS = "always"
    SIZE_MISMATCH = "size_mismatch"
    NEVER = "never"


class ParallelMode(str, Enum):
    SLURM = "slurm"
    MULTIPROCESS = "multiprocess"


def copy_src_to_dst(
    src_fs: fsspec.AbstractFileSystem,
    dst_fs: fsspec.AbstractFileSystem,
    src_file: str,
    dst_file: str,
    dry_run: bool = False,
):
    if dry_run:
        logging.info("Dry run copy: %s -> %s", src_file, dst_file)
    else:
        dst_parent_directory = os.path.dirname(dst_file)
        dst_fs.mkdirs(dst_parent_directory, exist_ok=True)
        with src_fs.open(src_file, "rb") as src_pointer, dst_fs.open(
            dst_file, "wb"
        ) as dst_pointer:
            shutil.copyfileobj(src_pointer, dst_pointer)


class CopyJob(submitit.helpers.Checkpointable):
    def __call__(
        self,
        src_fs_dict: dict,
        dst_fs_dict: dict,
        src_file: str,
        dst_file: str,
        dry_run: bool = False,
        validate_size: bool = True,
    ):
        src_fs = fsspec.AbstractFileSystem.from_dict(src_fs_dict)
        dst_fs = fsspec.AbstractFileSystem.from_dict(dst_fs_dict)
        copy_src_to_dst(src_fs, dst_fs, src_file, dst_file, dry_run=dry_run)
        if validate_size and not dry_run:
            src_size = src_fs.size(src_file)
            dst_size = dst_fs.size(dst_file)
            if src_size != dst_size:
                raise ValueError(
                    f"Mismatched sizes for src={src_file} dst={dst_file} {src_size} != {dst_size}"
                )
        return True


def main(
    src_dir: str,
    dst_dir: str,
    src_s3_profile: str | None = None,
    dst_s3_profile: str | None = None,
    n_workers: int = 16,
    cpus_per_task: int = 2,
    overwrite_mode: OverwriteMode = OverwriteMode.SIZE_MISMATCH,
    validate_size: bool = True,
    parallel_mode: ParallelMode = ParallelMode.MULTIPROCESS,
    dry_run: bool = False,
    job_dir: str = "jobs_parallel-copy",
    slurm_qos: str | None = None,
    slurm_time_hours: int = 72,
    slurm_memory: str = "0",
    wait: bool = True,
    wait_period: int = 5,
):
    logging.info("Starting parallell copy: %s -> %s", src_dir, dst_dir)
    logging.info("job_dir=%s", job_dir)
    logging.info(
        "Parallel=%s validate_size=%s overwrite_mode=%s qos=%s",
        parallel_mode,
        validate_size,
        overwrite_mode,
        slurm_qos,
    )
    if parallel_mode == ParallelMode.MULTIPROCESS:
        executor = submitit.LocalExecutor(folder=job_dir)
    elif parallel_mode == ParallelMode.SLURM:
        executor = submitit.SlurmExecutor(folder=job_dir)
        executor.update_parameters(
            time=slurm_time_hours * 60,
            ntasks_per_node=1,
            cpus_per_task=cpus_per_task,
            array_parallelism=n_workers,
            mem=slurm_memory,
            gpus_per_node=0,
        )
        if slurm_qos is not None:
            executor.update_parameters(qos=slurm_qos)
    else:
        raise ValueError("Invalid parallel mode")

    assert src_dir.endswith("/"), "src_dir must end with a /"
    assert dst_dir.endswith("/"), "dst_dir must end with a /"
    src_fs = get_fs(src_dir, s3_profile=src_s3_profile)
    dst_fs = get_fs(dst_dir, s3_profile=dst_s3_profile)

    src_dir = strip_s3_prefix(src_dir)
    dst_dir = strip_s3_prefix(dst_dir)
    logging.info("src: %s, dst: %s", src_dir, dst_dir)

    assert src_fs.isdir(src_dir), "src_dir must be a directory"
    if dst_fs.exists(dst_dir):
        assert dst_dir, "dst_dir must be a directory if it exists"
    else:
        dst_fs.mkdirs(dst_dir, exist_ok=True)

    files = src_fs.find(src_dir)
    logging.info("Files found to check for transfer: %s", len(files))
    jobs = []
    with executor.batch():
        for src_file in files:
            relative_src = src_file[len(src_dir) :]
            dst_file_path = os.path.join(dst_dir, relative_src)
            logging.debug("src: %s -> dst %s", src_file, dst_file_path)
            if dst_fs.exists(dst_file_path):
                if overwrite_mode == OverwriteMode.NEVER:
                    pass
                elif overwrite_mode == OverwriteMode.ALWAYS:
                    logging.info("copy: %s -> %s", src_file, dst_file_path)
                    job = executor.submit(
                        CopyJob(),
                        src_fs.to_dict(),
                        dst_fs.to_dict(),
                        src_file,
                        dst_file_path,
                        dry_run=dry_run,
                        validate_size=validate_size,
                    )
                    jobs.append(job)
                elif overwrite_mode == OverwriteMode.SIZE_MISMATCH:
                    if src_fs.size(src_file) != dst_fs.size(dst_file_path):
                        logging.info("copy: %s -> %s", src_file, dst_file_path)
                        job = executor.submit(
                            CopyJob(),
                            src_fs.to_dict(),
                            dst_fs.to_dict(),
                            src_file,
                            dst_file_path,
                            dry_run=dry_run,
                            validate_size=validate_size,
                        )
                        jobs.append(job)
                else:
                    raise ValueError("Unknown overwrite_mode")
            else:
                logging.info("copy: %s -> %s", src_file, dst_file_path)
                job = executor.submit(
                    CopyJob(),
                    src_fs.to_dict(),
                    dst_fs.to_dict(),
                    src_file,
                    dst_file_path,
                    dry_run=dry_run,
                    validate_size=validate_size,
                )
                jobs.append(job)
    if wait:
        while True:
            num_finished = sum(job.done() for job in jobs)
            logging.info("Total Jobs: %s Completed Jobs: %s", len(jobs), num_finished)
            if num_finished == len(jobs):
                break
            time.sleep(wait_period)
        output = [job.result() for job in jobs]
        if all(output):
            logging.info("All copies succeeded")
        else:
            logging.info("Some copies failed")
    else:
        logging.info("Not waiting for job to complete before exiting submit program")


if __name__ == "__main__":
    typer.run(main)
