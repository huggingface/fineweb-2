from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.readers import JsonlReader
from datatrove.io import DataFolder
# DATASET is not publicly available

SlurmPipelineExecutor(
    job_name="sailor2",
    pipeline=[
        JsonlReader("<REDACTED>", glob_pattern="thai/chunk_*.jsonl", default_metadata={"language": "th"}, doc_progress=True, file_progress=True, text_key="text"),
        JsonlWriter("/path/to/ref-datasets/monolingual/th/sea-commoncrawl", output_filename="${rank}.jsonl.gz", max_file_size=2*2**30)
    ],
    logging_dir="/path/to/logs/dataset_download_logs/th/sea-commoncrawl",
    randomize_start_duration=3 * 60,
    tasks=100,
    mem_per_cpu_gb=4,
    partition="partition",
    time="11:59:59",
).run()