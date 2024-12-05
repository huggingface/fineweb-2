from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.readers import ParquetReader

SlurmPipelineExecutor(
    job_name="dl_101b_arabicwords",
    pipeline=[
        ParquetReader("hf://datasets/ClusterlabAi/101_billion_arabic_words_dataset", glob_pattern="data/*.parquet", default_metadata={"language": "ar"}),
        JsonlWriter("/path/to/ref-datasets/monolingual/ar/101b_arabicwords", output_filename="${rank}.jsonl.gz", max_file_size=2*2**30)
    ],
    tasks=64,
    logging_dir="/path/to/logs/dataset_download_logs/ar/101b_arabicwords",
    randomize_start_duration=3 * 60,
    cpus_per_task=64,
    mem_per_cpu_gb=1,
    partition="normal",
    time="11:59:59",
    srun_args={ "environment" : "train" }
).run()