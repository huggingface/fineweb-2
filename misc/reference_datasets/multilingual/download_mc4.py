from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers import ParquetReader, JsonlReader
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
    import os.path
    if "validation." in path:
        return {}
    return {
        "text": data.pop(self.text_key, ""),
        "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
        "media": data.pop("media", []),
        "metadata": {"language": os.path.basename(path).split(".")[0].split("-")[1]} | data.pop("metadata", {}) | data,
        # remaining data goes into metadata
    }


SlurmPipelineExecutor(
    job_name="mc4",
    pipeline=[
        JsonlReader("hf://datasets/allenai/c4/multilingual",
                    glob_pattern="c4-*.*.json.gz", adapter=adapter),
        JsonlWriter("/path/to/ref-datasets/mc4",
                    output_filename="${language}" + "/${rank}.jsonl.gz")
    ],
    tasks=300,
    # workers=50,
    mem_per_cpu_gb=4,
    logging_dir="/path/to/logs/multilingual/copy/mc4",
    partition="partition",
    randomize_start_duration=10 * 60,
    time="20:00:00"
).run()
