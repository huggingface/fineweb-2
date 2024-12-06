import argparse
import os
import subprocess
import tempfile

from loguru import logger

USER=os.environ["USER"]


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

parser = argparse.ArgumentParser("Launch the original english evals for a set of checkpoints.")

parser.add_argument(
    "language", type=str, help="Language to run evals for. Example: zh"
)
parser.add_argument(
    "checkpoints", type=str, help="Checkpoints to run evals for. Example: 0,1,2", default=0
)
parser.add_argument(
    "--offline-datasets", action="store_true", help="Turns off datasets downloading"
)
parser.add_argument("--qos", type=str, default="normal", help="qos to use")
parser.add_argument("--time_limit", type=str, default="01:20:00", help="slurm time limit. 15:00 by default")
parser.add_argument("--parallel", "-p",type=int, default=100, help="How many eval tasks to run simultaneously")
parser.add_argument("--logging_dir", type=str, default="/path/to/eval-results", help="Repo to push results to")
parser.add_argument("-d", help="dependency job", type=str, default=None)
parser.add_argument("--overwrite", "-ow", action="store_true",
                    help="Overwrite existing eval results. Will skip completed checkpoints by default")
parser.add_argument("--tasks", type=str, default="early-signals", help="Tasks to run. Example: all,m3exam")
parser.add_argument("--tokenizer", type=str, default="google/gemma-7b", help="Tokenizer to use for the model")

if __name__ == "__main__":
    args = parser.parse_args()
    job_id = None
    model_name = f"dummy-{args.language}-"
    selected_checkpoints = args.checkpoints.split(",")
    bash_ckpts_list = "(" + " ".join(f'"{item}"' for item in sorted(map(int, selected_checkpoints), reverse=True)) + ")"
    os.makedirs(f"/path/to/eval-logs/{model_name}/{args.language}", exist_ok=True)
    deps = []
    if args.d:
        deps.append(f"afterok:{args.d}")
    if job_id:
        deps.append(f"afterany:{job_id}")

    launch_script = f"""#!/bin/bash
#SBATCH --job-name=eval-{model_name}-{args.language}
#SBATCH --tasks=1
#SBATCH --partition=partition
#SBATCH --qos={args.qos}
#SBATCH --array=0-{len(selected_checkpoints)-1}%{args.parallel}
#SBATCH --time={args.time_limit}
#SBATCH --cpus-per-task=4
#SBATCH --output=/path/to/logs/train/multilingual/eval-logs/{model_name}/{args.language}/eval-%A_%a.out
#SBATCH --error=/path/to/logs/train/multilingual/eval-logs/{model_name}/{args.language}/eval-%A_%a.out
{"#SBATCH --dependency=" + ",".join(deps) if deps else ""}
#SBATCH --requeue
###########################################
# [BEGINING] ADAPT TO YOUR ENVIRONMENT
source /admin/home/{USER}/.bashrc
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/miniconda3/envs/exp/


LIGHTEVAL_FOLDER=/path/to/ml-lighteval
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
export TMPDIR=/scratch/{USER}/{model_name}/{args.language}
mkdir -p $TMPDIR
CHECKPOINTS_LIST={bash_ckpts_list}
NSTEP=$((SLURM_ARRAY_TASK_ID))
STEP=${{CHECKPOINTS_LIST[$NSTEP]}}

launch_args="$LIGHTEVAL_FOLDER/run_evals_accelerate.py --model_args='dummy,name=dummy-{args.language}-/${{STEP}},tokenizer={args.tokenizer}' --max_samples=1000 --custom_tasks=lighteval.community_tasks.multilingual.configs.{args.language} --tasks={args.tasks} --save_results --logging_dir={args.logging_dir}"
sleep $((RANDOM % 60))
srun -u bash -c "python3 -u ${{launch_args}}" """

    launched_id = launch_slurm_job(launch_script)
    logger.success(f"{model_name} evals launched with id={launched_id}. Logs: /path/to/logs/train/multilingual/eval-logs/{model_name}/{args.language}")
    job_id = launched_id