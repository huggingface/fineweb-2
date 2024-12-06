import os
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime

from nanotron.logging import human_format
from nanotron.models.llama import LlamaConfig

from datatrove.io import get_datafolder
from nanotron.config import DatasetStageArgs, NanosetDatasetsArgs, S3UploadArgs


# Paths
LOCAL_TMP_PATH_ON_NODE = f"/scratch/{os.environ.get('USER')}"
LAUNCH_CONFIGS_PATH = f"path/to/launch-configs"

# Executables
NANOTRON_RUN_TRAIN_SCRIPT = f"path/to/run_train.py"
S5CMD_PATH = "path/to/s5cmd"

S3_CHECKPOINTS_PREFIX = "path/to/where_to_save_checkpoints"

# Logging parameters
LOGS_PATH = f"path/to/slurm-logs"
REPO_ID = f"id of the repo to use for logging"
PROJECT = "name of the project"
EMAIL = "email to send notifications to"

# Resources parameters
NUM_GPUS = 8
NUM_CPUS_IN_NODE = 88
CPUS_PER_GPU = NUM_CPUS_IN_NODE // NUM_GPUS


model_config = LlamaConfig(
    # Config for a 1.46B model
    bos_token_id=1,
    eos_token_id=2,
    hidden_act="silu",
    hidden_size=2048,
    initializer_range=0.02,
    intermediate_size=8192,
    max_position_embeddings=2048,
    num_attention_heads=32,
    num_hidden_layers=14,
    num_key_value_heads=32,
    pretraining_tp=1,
    rms_norm_eps=1e-05,
    rope_scaling=None,
    tie_word_embeddings=True,
    use_cache=True,
    vocab_size=256008,  # XGLM tokenizer rounded to next multiple of 8
)


num_params = human_format(
    model_config.vocab_size * model_config.hidden_size +
    model_config.num_hidden_layers
    * (
            3 * model_config.hidden_size * model_config.intermediate_size
            + 4 * model_config.hidden_size * model_config.hidden_size
    )
).replace(".", "p")

print(f"Model has {num_params} parameters")


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
        return subprocess.check_output(["sbatch", *args, f.name]).decode("utf-8").split()[-1]


if __name__ == "__main__":
    import argparse
    from dataclasses import fields, is_dataclass

    from nanotron.config import get_config_from_file

    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="dataset folder", type=str)
    parser.add_argument("run_name", help="run name", type=str)
    parser.add_argument("language", help="language", type=str)
    parser.add_argument("-d", help="dependency job", type=str, default=None)
    parser.add_argument("--seed", help="seed", type=int, default=6)
    parser.add_argument("--train_steps", "-ts", help="training steps. Total_toks=seq_len*steps*micro_bs*batch_accum_per_replica*dp_size", type=int, default=14000)
    parser.add_argument("--priority", "--qos", "-p", help="qos to use", type=str, default="normal")
    args = parser.parse_args()
    SEED = args.seed


    dataset_name = run_name = args.run_name.replace(" ", "_")

    # Specific name for this run (checkpoints/logs/tensorboard)
    RUN = f"{num_params}-{dataset_name}-seed-{SEED}"

    df = get_datafolder(f"{S3_CHECKPOINTS_PREFIX}/{RUN}")
    if df.exists("latest.txt") and df.cat_file("latest.txt") == bytes(str(args.train_steps), "utf-8"):
        print(f"Not launching as latest checkpoint is already {args.train_steps} steps")
        sys.exit(0)

    import torch

    from nanotron.config import (
        CheckpointsArgs,
        Config,
        DataArgs,
        GeneralArgs,
        LlamaConfig,
        LoggingArgs,
        LRSchedulerArgs,
        ModelArgs,
        OptimizerArgs,
        ParallelismArgs,
        RandomInit,
        TokenizerArgs,
        TokensArgs,
        AdamWOptimizerArgs,
    )

    def print_differences(target, updates):
        if not is_dataclass(target) or not is_dataclass(updates):
            raise ValueError("Both target and updates should be dataclass instances")

        for field in fields(target):
            update_value = getattr(updates, field.name)

            if update_value is not None:
                if is_dataclass(update_value):
                    print_differences(getattr(target, field.name), update_value)
                else:
                    target_value = getattr(target, field.name)
                    if update_value != target_value:
                        if update_value.__class__.__module__ != "builtins":
                            continue
                        print(f"{field.name}: {target_value} -> {update_value}")

    data = [
        DatasetStageArgs(
            name="Training Stage",
            start_training_step=1,
            data=DataArgs(
                seed=SEED,
                num_loading_workers=0,
                dataset=NanosetDatasetsArgs(
                    dataset_folder=args.data if not args.data.startswith("s3://") else f"{LOCAL_TMP_PATH_ON_NODE}/dataset/{RUN}/",
                    dataset_weights=None,
                )
            )
        ),
    ]

    general = GeneralArgs(
        project=PROJECT,
        run=RUN,
        ignore_sanity_checks=True,
        seed=SEED,
    )

    checkpoints = CheckpointsArgs(
        checkpoints_path=Path(f"{LOCAL_TMP_PATH_ON_NODE}/checkpoints/{RUN}"),
        checkpoints_path_is_shared_file_system=False,
        checkpoint_interval=500,
        save_initial_state=True,
    )

    parallelism = ParallelismArgs(
        dp=64,
        pp=1,
        tp=1,
        pp_engine="1f1b",
        tp_mode="REDUCE_SCATTER",
        tp_linear_async_communication=True,
    )
    # num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
    # parallelism.dp=int(num_nodes*8//parallelism.pp//parallelism.tp),  # How many remaining GPU when taking into account PP, TP and 8 GPUs per node

    tokens = TokensArgs(
        batch_accumulation_per_replica=4,
        micro_batch_size=4,
        sequence_length=2048,
        train_steps=args.train_steps,
        val_check_interval=-1,
    )

    model = ModelArgs(
        model_config=model_config,
        make_vocab_size_divisible_by=1,
        init_method=RandomInit(
            std=0.02,
            # std=1
            # / math.sqrt(model_config.hidden_size)  # 0.01275  # Basically 1/sqrt(N),
            # path="/fsx/shared-falcon-180B/brrr-falcon-180B"
        ),
        dtype=torch.bfloat16,
    )

    logging = LoggingArgs(
        # 'debug', 'info', 'warning', 'error', 'critical' and 'passive'
        log_level="info",
        log_level_replica="info",
        iteration_step_info_interval=1,
    )

    optimizer = OptimizerArgs(
        accumulate_grad_in_fp32=True,
        clip_grad=1.0,
        weight_decay=0.1,
        zero_stage=0,
        learning_rate_scheduler=LRSchedulerArgs(
            learning_rate=3e-4,
            lr_warmup_steps=500,
            lr_warmup_style="linear",
            lr_decay_style="cosine",
            # lr_decay_steps=10000-500,  # Keeping it to 10k for comparision for now
            min_decay_lr=3.0e-5
        ),
        optimizer_factory=AdamWOptimizerArgs(
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1.0e-8,
            torch_adam_is_fused=True,
        ),
    )

    tokenizer = TokenizerArgs(
        tokenizer_name_or_path="google/gemma-7b",
    )

    s3_upload = S3UploadArgs(
        upload_s3_path=f"{S3_CHECKPOINTS_PREFIX}/{RUN}",
        remove_after_upload=True,
        s5cmd_numworkers=16,
        s5cmd_concurrency=5,
        s5cmd_path=S5CMD_PATH,
    )

    config = Config(
        general=general,
        checkpoints=checkpoints,
        parallelism=parallelism,
        model=model,
        tokenizer=tokenizer,
        logging=logging,
        tokens=tokens,
        optimizer=optimizer,
        data_stages=data,
        profiler=None,
        s3_upload=s3_upload,
        lighteval=None,
    )

    NODES = 8
    #### DEBUG MODE
    if os.environ.get("DEBUG_MODE", "0") != "0":
        print("##### WARNING DEBUG MODE #####")
        config.parallelism.dp = 2
        config.parallelism.pp = 2
        config.parallelism.tp = 2
        config.tokens.micro_batch_size = 3
        config.tokens.batch_accumulation_per_replica = 2
        config.checkpoints.save_initial_state = True
        NODES = 1

    # Sanity check that we can load, save to YAML and reload the config
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f"{LAUNCH_CONFIGS_PATH}/{run_name}", exist_ok=True)
    config_path_yaml = f"{LAUNCH_CONFIGS_PATH}/{run_name}/{timestamp}.yaml"
    config.save_as_yaml(config_path_yaml)
    config2 = get_config_from_file(config_path_yaml, config_class=Config)
    print_differences(config, config2)

    os.makedirs(f"{LOGS_PATH}/{run_name}", exist_ok=True)

    dataset_download_cmd =  "" if not args.data.startswith("s3://") else f"srun --ntasks-per-node=1 rm -rf {LOCAL_TMP_PATH_ON_NODE}/dataset\nsrun --ntasks-per-node=1 s5cmd cp '{args.data.removesuffix('/')}/*' {LOCAL_TMP_PATH_ON_NODE}/dataset/{RUN}/"
    job_name = f"{run_name}-{SEED}"

    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={NODES}
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task={NUM_CPUS_IN_NODE}
#SBATCH --gres=gpu:{NUM_GPUS}
#SBATCH --partition=hopper-prod
#SBATCH --output={LOGS_PATH}/{run_name}/train-{timestamp}-%x-%j
# #SBATCH --array=1-1%1
#SBATCH --qos={args.priority}
#SBATCH --begin=now+0minutes
#SBATCH --mail-type=ALL
#SBATCH --mail-user={EMAIL}
#SBATCH --requeue
{"#SBATCH --dependency=afterok:" + args.d if args.d else ""}

###########################################
# [BEGINING] ADAPT TO YOUR ENVIRONMENT


# [END] ADAPT TO YOUR ENVIRONMENT
###########################################


set -x -e

##### TO UPDATE #####


##### END TO UPDATE ######

echo "START TIME: $(date)"
secs_to_human(){{
    echo "$(( ${{1}} / 3600 )):$(( (${{1}} / 60) % 60 )):$(( ${{1}} % 60 ))"
}}
start=$(date +%s)
echo "$(date -d @${{start}} "+%Y-%m-%d %H:%M:%S"): ${{SLURM_JOB_NAME}} start id=${{SLURM_JOB_ID}}\n"

{dataset_download_cmd}

# SLURM stuff
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((1024 + RANDOM % 64511))
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

export TMPDIR={LOCAL_TMP_PATH_ON_NODE}
export CUDA_DEVICE_MAX_CONNECTIONS="1"

module load cuda/12.1

echo go $COUNT_NODE
echo $HOSTNAMES

##### MOVE TO YAML ######

CMD=" \
    {NANOTRON_RUN_TRAIN_SCRIPT} \
    --config-file {config_path_yaml}
    "

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node {NUM_GPUS} \
    --nnodes $COUNT_NODE \
    --rdzv-backend c10d \
    --rdzv-endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv-id $SLURM_JOB_ID \
    --node_rank $SLURM_PROCID \
    --role $SLURMD_NODENAME: \
    --max_restarts 0 \
    --tee 3 \
    "

# Wait a random number between 0 and 1000 (milliseconds) to avoid too many concurrent requests to the hub
random_milliseconds=$(( RANDOM % 1001 ))
sleep_time=$(bc <<< "scale=3; $random_milliseconds / 1000")
echo "Sleeping for $sleep_time seconds..."
sleep $sleep_time

launch_args="srun $SRUN_ARGS -u bash -c $LAUNCHER --node_rank $SLURM_PROCID --role $SLURMD_NODENAME: $CMD"

srun $SRUN_ARGS -u bash -c "$LAUNCHER --node_rank $SLURM_PROCID --role $SLURMD_NODENAME: $CMD"


echo "END TIME: $(date)"

{
    "" if not args.data.startswith("s3://") else f"srun --ntasks-per-node=1 rm -rf {LOCAL_TMP_PATH_ON_NODE}/dataset/{RUN}/"
}
"""
    id = launch_slurm_job(sbatch_script)
    log_path = f"{LOGS_PATH}/{run_name}/train-{timestamp}-{job_name}-{id}"
    print(f"Launched with Slurm job id={id}")
    print(f"To view the logs, use the command: tail -f {log_path}")
