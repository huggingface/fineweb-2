import argparse
from datetime import datetime
import os
import re
import subprocess
import tempfile
from typing import Optional

from fsspec.core import url_to_fs
import itertools
from datatrove.io import get_datafolder
from loguru import logger


EVAL_LOGS_PATH = f"/path/to/eval-logs"
CPUS_PER_NODE = 88
GPUS_PER_NODE = 8
PARTITION = "partition"
NODES = 1


def parse_date(date_string: Optional[str]) -> Optional[datetime]:
    if date_string is None:
        return None
    try:
        return datetime.strptime(date_string, "%d-%m-%Y %H:%M:%S")
    except ValueError:
        raise ValueError("Invalid date format. Use 'DD-MM-YYYY HH:MM:SS'")


def checkpoint_exists(logging_dir: str, model_name: str, checkpoint: str, reference_date: Optional[datetime]) -> bool:
    fs, path = url_to_fs(logging_dir)
    try:
        result_files = fs.glob(f"{path}/results/{model_name}/{checkpoint}/results_*.json")
    except FileNotFoundError:
        result_files = []

    if len(result_files) == 0:
        return False

    if reference_date is None:
        return True

    timestamps = [datetime.strptime(re.search(r'results_(.*)\.json$', f).group(1), "%Y-%m-%dT%H-%M-%S.%f") for f in
                  result_files]
    return any(timestamp > reference_date for timestamp in timestamps)


def launch_slurm_job(launch_file_contents, *args):
    """
        Small helper function to save a sbatch script and call it.
    Args:
        launch_file_contents: Contents of the sbatch script
        *args: any other arguments to pass to the sbatch command

    Returns: the id of the launched slurm job

    """
    with tempfile.NamedTemporaryFile("w") as f:
        f.write(launch_file_contents)
        f.flush()
        try:
            return subprocess.check_output(["sbatch", *args, f.name]).decode("utf-8").split()[-1]
        except Exception as e:
            print(launch_file_contents, flush=True)
            raise e


def get_checkpoints_to_run(s3_path: str, model_name: str, checkpoints: str, logging_dir: str, overwrite: bool = False,
                           after_date: Optional[str] = None):
    reference_date = parse_date(after_date)
    df = get_datafolder(s3_path)
    try:
        avail_checkpoints = [i for i in sorted(df.ls("", detail=False)) if i != "latest.txt"]
    except FileNotFoundError:
        logger.error(f"No checkpoints found in {s3_path}")
        avail_checkpoints = []
    logger.info(f"Found {len(avail_checkpoints)} checkpoints")
    selected_checkpoints = checkpoints.split(",") if checkpoints != "all" else avail_checkpoints
    not_found_checkpoints = [ckpt for ckpt in selected_checkpoints if ckpt not in avail_checkpoints]
    if len(not_found_checkpoints) > 0:
        raise ValueError(f"Checkpoints not found in \"{s3_path}\": {not_found_checkpoints}")

    if not overwrite:
        # remove completed checkpoints
        completed_checkpoints = [
            ckpt for ckpt in selected_checkpoints
            if checkpoint_exists(logging_dir, model_name, ckpt, reference_date)
        ]
        completed = len(completed_checkpoints)
        selected_checkpoints = list(set(selected_checkpoints) - set(completed_checkpoints))
        if completed:
            logger.info(f"Skipping {completed} already evaluated checkpoints.")
    return selected_checkpoints


parser = argparse.ArgumentParser("Launch evals for a set of checkpoints.")

parser.add_argument(
    "model_name", type=str,
    help="Model name on s3. Example: 1p46G-control-english-fw-ft-bl-28BT-seed-6. Use commas for multiple models"
)
parser.add_argument(
    "language", type=str, help="Language to run evals for. Example: zh"
)
parser.add_argument(
    "--s3_prefix", type=str, help="s3://path/to/models/ by default",
    default="s3://path/to/models/"
)
parser.add_argument(
    "--checkpoints", "-ckpts", type=str, help="Comma separated list of checkpoints to run, or \"all\"",
    default="all"
)
parser.add_argument(
    "--model-template", type=str, help="Template to use for the model name",
    default="{model_name}"
    # default="{model_name}-{language}-29BT-seed-{seed}"
)

parser.add_argument("--tasks", type=str, help="Comma separated list of tasks to run, or \"all\"",
                    default="early-signals")
parser.add_argument(
    "--offline-datasets", action="store_true", help="Turns off datasets downloading", default=True
)
parser.add_argument(
    "--seed", help="Defines seeds to use in model template. Comma separated list of seeds", default="6"
)
parser.add_argument("--qos", type=str, default="normal", help="qos to use")
parser.add_argument("--time_limit", type=str, default="1:50:00", help="slurm time limit. 1:50:00 by default")
parser.add_argument("--parallel", "-p", type=int, default=5, help="How many eval tasks to run simultaneously")
# parser.add_argument("--batch_size", "-bs", type=int, default=8, help="Batch size")
parser.add_argument("--gpus", "-g", type=int, default=GPUS_PER_NODE, help="How many gpus to use")
parser.add_argument("--logging_dir", type=str, default="s3://path/to/evals/results",
                    help="Repo to push results to")
parser.add_argument("-d", help="dependency job", type=str, default=None)
parser.add_argument("--overwrite", "-ow", action="store_true", default=False,
                    help="Overwrite existing eval results. Will skip completed checkpoints by default")
parser.add_argument("--after-date", type=str, default=None,
                    help="Only consider checkpoints newer than this date (DD-MM-YYYY HH:MM:SS)")
parser.add_argument("--job-prefix", type=str, default="", help="Prefix to add to the job name")

if __name__ == "__main__":
    args = parser.parse_args()
    job_id = None
    for model_name, seed in itertools.product(args.model_name.split(","), args.seed.split(",")):
        model_name = args.model_template.format(model_name=model_name, language=args.language, seed=seed)
        s3_path = args.s3_prefix.removesuffix("/") + "/" + model_name if not model_name.startswith(
            "s3://") else model_name
        selected_checkpoints = get_checkpoints_to_run(s3_path, model_name, args.checkpoints, args.logging_dir,
                                                      overwrite=args.overwrite, after_date=args.after_date)
        logger.info(f"Found {len(selected_checkpoints)} checkpoints for {model_name}")
        if not selected_checkpoints:
            print("No checkpoints to run.")
            continue
        bash_ckpts_list = "(" + " ".join(
            f'"{item}"' for item in sorted(map(int, selected_checkpoints), reverse=True)) + ")"
        os.makedirs(f"{EVAL_LOGS_PATH}/{model_name}/{args.language}", exist_ok=True)

        n_cpus = CPUS_PER_NODE // args.gpus

        # Write the lightevalconf.yml file
        with open(f"{EVAL_LOGS_PATH}/{model_name}/{args.language}.yml", "wt") as f:
            f.write(f"""batch_size: {4 if args.language == "zh" else (6 if args.language in ("ar", "sw") else 8)}
checkpoints_path: null
generation: null
logging:
  logging_dir: {args.logging_dir}
  save_details: true
  save_results: true
  save_to_tensorboard: false
  tensorboard_metric_prefix: e
parallelism:
  dp: {args.gpus}
  expert_parallel_size: 1
  pp: 1
  pp_engine: 1f1b
  tp: 1
  tp_linear_async_communication: false
  tp_mode: ALL_REDUCE
tasks:
  custom_tasks: lighteval.community_tasks.multilingual.configs.{args.language}
  dataset_loading_processes: {n_cpus}
  max_samples: 1000
  multichoice_continuations_start_space: null
  no_multichoice_continuations_start_space: null
  num_fewshot_seeds: null
  tasks: {args.tasks}""")

        deps = []
        if args.d:
            deps.append(f"afterok:{args.d}")
        if job_id:
            deps.append(f"afterany:{job_id}")

        launch_script = f"""#!/bin/bash
#SBATCH --job-name={args.job_prefix}eval-{model_name}
#SBATCH --nodes={NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --partition={PARTITION}
{f'#SBATCH --qos={args.qos}' if args.qos else ''}
#SBATCH --array=0-{len(selected_checkpoints) - 1}%{args.parallel}
#SBATCH --gres=gpu:{args.gpus}
#SBATCH --time={args.time_limit}
#SBATCH --cpus-per-task={CPUS_PER_NODE}
#SBATCH --output={EVAL_LOGS_PATH}/{model_name}/{args.language}/eval-%A_%a.out
#SBATCH --error={EVAL_LOGS_PATH}/{model_name}/{args.language}/eval-%A_%a.out
{"#SBATCH --dependency=" + ",".join(deps) if deps else ""}
#SBATCH --requeue
###########################################
# [BEGINING] ADAPT TO YOUR ENVIRONMENT
source /path/to/.bashrc
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/miniconda3/envs/exp/

BRRR_FOLDER=/path/to/brrr
# Ensure cache is on fsx not on admin
export HUGGINGFACE_HUB_CACHE=/path/to/.cache/huggingface
export HF_DATASETS_CACHE=/path/to/.cache/huggingface
export HF_MODULES_CACHE=/path/to/.cache/huggingface
export HF_HOME=/path/to/.cache/huggingface
export HF_DATASETS_OFFLINE={1 if args.offline_datasets else 0}

# [END] ADAPT TO YOUR ENVIRONMENT
###########################################


set -x -e
echo "START TIME: $(date)"
echo python3 version = `python3 --version`

# SLURM stuff
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=6000
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export CUDA_DEVICE_MAX_CONNECTIONS="1"

module load cuda/12.1

echo go $COUNT_NODE
echo $HOSTNAMES
CHECKPOINTS_LIST={bash_ckpts_list}
NSTEP=$((SLURM_ARRAY_TASK_ID))
STEP=${{CHECKPOINTS_LIST[$NSTEP]}}


export TMPDIR=/scratch/USER/{model_name}/{args.language}/$STEP
mkdir -p $TMPDIR

LOCAL_DOWNLOAD_CHECKPOINT_FOLDER=/scratch/USER/checkpoint/{model_name}/$STEP
# Copying checkpoint from s3 to the node on node
mkdir -p $LOCAL_DOWNLOAD_CHECKPOINT_FOLDER
s5cmd cp --exclude "optimizer/*" {s3_path}/$STEP/* $LOCAL_DOWNLOAD_CHECKPOINT_FOLDER

torch_dist_args="--nproc_per_node {args.gpus} \\
    --nnodes $COUNT_NODE \\
    --max_restarts 0 \\
    --tee 3 \\
    --node_rank $SLURM_PROCID \\
    --role $SLURMD_NODENAME: "

launch_args="$torch_dist_args $BRRR_FOLDER/run_evals_nanotron.py \\
    --checkpoint-config-path ${{LOCAL_DOWNLOAD_CHECKPOINT_FOLDER}}/config.yaml --lighteval-override {EVAL_LOGS_PATH}/{model_name}/{args.language}.yml"

sleep $((RANDOM % 60))
srun -u bash -c "python3 -u -m torch.distributed.run ${{launch_args}}" """
        launched_id = launch_slurm_job(launch_script)
        logger.success(
            f"{model_name} evals with {args.gpus} gpus launched with id={launched_id}. Logs: {EVAL_LOGS_PATH}/{model_name}/{args.language}")
        job_id = launched_id
"""
RUN MANUALLY:
conda activate exp
LOCAL_DOWNLOAD_CHECKPOINT_FOLDER=/scratch/$USER/checkpoint/modeltest
mkdir -p $LOCAL_DOWNLOAD_CHECKPOINT_FOLDER
s5cmd cp --exclude "optimizer/*" {s3_path}/$STEP/* $LOCAL_DOWNLOAD_CHECKPOINT_FOLDER
source /etc/profile.d/modules.sh

export HF_HOME=/path/to/.cache/huggingface
export HF_DATASETS_OFFLINE=1
module load cuda/12.1

python3 -u -m torch.distributed.run --standalone /path/to/brrr/run_evals_nanotron.py --checkpoint-config-path ${LOCAL_DOWNLOAD_CHECKPOINT_FOLDER}/config.yaml --lighteval-config /path/to/configs/testmlevals.yml
"""
