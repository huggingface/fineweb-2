from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.readers import ParquetReader, JsonlReader
from datatrove.io import DataFolder

SlurmPipelineExecutor(
    job_name="odaigen_hindi",
    pipeline=[
        JsonlReader("hf://datasets/Hindi-data-hub/odaigen_hindi_pre_trained_sp", glob_pattern="**/*.json", default_metadata={"language": "hi"}, doc_progress=True, file_progress=True, text_key="content"),
        JsonlWriter("/path/to/ref-datasets/monolingual/hi/odaigen_hindi", output_filename="${rank}.jsonl.gz", max_file_size=2*2**30)
    ],
    logging_dir="/path/to/logs/dataset_download_logs/hi/odaigen_hindi",
    randomize_start_duration=3 * 60,
    tasks=100,
    mem_per_cpu_gb=1,
    partition="partition",
    time="11:59:59",
).run()