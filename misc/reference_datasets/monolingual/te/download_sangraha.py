from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter

SlurmPipelineExecutor(
    pipeline=[
        ParquetReader("hf://datasets/ai4bharat/sangraha", glob_pattern="synthetic/tel_Telu/*.parquet", id_key="doc_id", default_metadata={"language": "te", "subset": "synthetic"}),
        ParquetReader("hf://datasets/ai4bharat/sangraha", glob_pattern="verified/tel/*.parquet", id_key="doc_id", default_metadata={"language": "te", "subset": "verified"}),
        ParquetReader("hf://datasets/ai4bharat/sangraha", glob_pattern="unverified/tel/*.parquet", id_key="doc_id", default_metadata={"language": "te", "subset": "unverified"}),
        JsonlWriter("/path/to/ref-datasets/monolingual/te/sangraha",
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
    logging_dir="/path/to/logs/dataset_download_logs/te/sangraha",
).run()
from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.readers import ParquetReader
from datatrove.io import DataFolder

SlurmPipelineExecutor(
    job_name="sangraha",
    pipeline=[
        ParquetReader("hf://datasets/ai4bharat/sangraha", glob_pattern="**/tel/*.parquet", default_metadata={"language": "te"}, doc_progress=True, file_progress=True, text_key="text"),
        ParquetReader("hf://datasets/ai4bharat/sangraha", glob_pattern="**/tel_Telu/*.parquet", default_metadata={"language": "te"}, doc_progress=True, file_progress=True, text_key="text"),
        JsonlWriter("/path/to/ref-datasets/monolingual/te/sangraha", output_filename="${rank}.jsonl.gz", max_file_size=2*2**30)
    ],
    logging_dir="/path/to/logs/dataset_download_logs/te/sangraha",
    randomize_start_duration=3 * 60,
    tasks=100,
    mem_per_cpu_gb=4,
    partition="partition",
    time="11:59:59",
).run()