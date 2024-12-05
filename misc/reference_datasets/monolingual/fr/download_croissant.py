from typing import Callable
from datatrove.io import DataFolderLike, DataFileLike
from datatrove.pipeline.readers.base import BaseDiskReader
from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.writers import JsonlWriter

class ArrowReader(BaseDiskReader):
    name = "📒 Arrow"
    _requires_dependencies = ["pyarrow"]

    def __init__(
        self,
        data_folder: DataFolderLike,
        paths_file: DataFileLike | None = None,
        limit: int = -1,
        skip: int = 0,
        batch_size: int = 1000,
        read_metadata: bool = True,
        file_progress: bool = False,
        doc_progress: bool = False,
        adapter: Callable = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict = None,
        recursive: bool = True,
        glob_pattern: str | None = None,
        shuffle_files: bool = False,
    ):
        super().__init__(
            data_folder,
            paths_file,
            limit,
            skip,
            file_progress,
            doc_progress,
            adapter,
            text_key,
            id_key,
            default_metadata,
            recursive,
            glob_pattern,
            shuffle_files,
        )
        self.batch_size = batch_size
        self.read_metadata = read_metadata

    def read_file(self, filepath: str):
        import pyarrow as pa

        with self.data_folder.open(filepath, "rb") as f:
            reader = pa.ipc.open_stream(f)
            li = 0
            columns = [self.text_key, self.id_key] if not self.read_metadata else None
            documents = []
            with self.track_time("table"):
                df = reader.read_pandas(categories=columns)
                for _, row in df.iterrows():
                    document = self.get_document_from_dict(row.to_dict(), filepath, li)
                    if not document:
                        continue
                    documents.append(document)
                    li += 1
            yield from documents


SlurmPipelineExecutor(
    pipeline=[
        ArrowReader("hf://datasets/croissantllm/croissant_dataset", glob_pattern="french_*/train/*.arrow", default_metadata={"language": "fr"}),
        JsonlWriter("/path/to/ref-datasets/monolingual/fr/croissant",
                    output_filename="${rank}.jsonl.gz", max_file_size=2*2**30)
    ],
    tasks=32,
    randomize_start_duration=3 * 60,
    time="11:59:59",
    job_name="dl_croissant",
    cpus_per_task=64,
    mem_per_cpu_gb=1,
    partition="partition",
    srun_args={"environment": "train"},
    logging_dir="/path/to/logs/dataset_download_logs/fr/croissant",
).run()
