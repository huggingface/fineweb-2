from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter

SlurmPipelineExecutor(
    pipeline=[
        ParquetReader("hf://datasets/ai4bharat/sangraha", glob_pattern="synthetic/hin_Deva/*.parquet", id_key="doc_id", default_metadata={"language": "hi", "subset": "synthetic"}),
        ParquetReader("hf://datasets/ai4bharat/sangraha", glob_pattern="verified/hin/*.parquet", id_key="doc_id", default_metadata={"language": "hi", "subset": "verified"}),
        ParquetReader("hf://datasets/ai4bharat/sangraha", glob_pattern="unverified/hin/*.parquet", id_key="doc_id", default_metadata={"language": "hi", "subset": "unverified"}),
        JsonlWriter("/path/to/ref-datasets/monolingual/hi/sangraha",
                    output_filename="${rank}.jsonl.gz", max_file_size=2*2**30)
    ],
    tasks=32,
    randomize_start_duration=3 * 60,
    time="11:59:59",
    job_name="dl_sangraha",
    cpus_per_task=64,
    mem_per_cpu_gb=1,
    partition="partition",
    srun_args={"environment": "train"},
    logging_dir="/path/to/logs/dataset_download_logs/hi/sangraha",
).run()