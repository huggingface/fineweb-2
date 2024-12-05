from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter

SlurmPipelineExecutor(
    pipeline=[
        JsonlReader("hf://datasets/liwu/MNBVC", glob_pattern="**/*.jsonl.gz", default_metadata={"language": "zh"}),
        JsonlWriter("/path/to/ref-datasets/monolingual/zh/mnbvc",
                    output_filename="${rank}.jsonl.gz", max_file_size=2*2**30)
    ],
    tasks=64,
    randomize_start_duration=3 * 60,
    time="11:59:59",
    job_name="dl_mnbvc",
    cpus_per_task=64,
    mem_per_cpu_gb=1,
    partition="partition",
    srun_args={"environment": "train"},
    logging_dir="/path/to/logs/dataset_download_logs/zh/mnbvc",
).run()
