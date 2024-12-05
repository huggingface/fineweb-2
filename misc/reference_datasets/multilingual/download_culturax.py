from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter


def adapter(self, data: dict, path: str, id_in_file: int | str):
    """
    The default data adapter to adapt input data into the datatrove Document format

    Args:
        data: a dictionary with the "raw" representation of the data
        path: file path or source for this sample
        id_in_file: its id in this particular file or source

    Returns: a dictionary with text, id, media and metadata fields

    """
    return {
        "text": data.pop(self.text_key, ""),
        "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
        "media": data.pop("media", []),
        "metadata": {"language": path.split("/")[0]} | data.pop("metadata", {}) | data,
        # remaining data goes into metadata
    }


SlurmPipelineExecutor(
    job_name="culturax",
    pipeline=[
        ParquetReader("hf://datasets/uonlp/CulturaX", glob_pattern="*/*.parquet", adapter=adapter),
        JsonlWriter("/path/to/ref-datasets/culturax",
                    output_filename="${language}" + "/${rank}.jsonl.gz")
    ],
    tasks=1000,
    mem_per_cpu_gb=4,
    logging_dir="/path/to/logs/multilingual/copy/culturax",
    partition="partition",
    randomize_start_duration=3 * 60,
    time="20:00:00"
).run()
