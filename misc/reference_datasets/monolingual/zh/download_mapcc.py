import gzip
import itertools
import json
import fsspec
from datatrove.executor import LocalPipelineExecutor, SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers import JsonlWriter
import orjson

class ConcatenatedFileStream:
    def __init__(self, filepaths):
        self.filepaths = filepaths
        self.file_index = 0
        self.current_file = None
        self._open_next_file()

    def _open_next_file(self):
        if self.current_file:
            self.current_file.close()
        if self.file_index < len(self.filepaths):
            print(f"opening {self.filepaths[self.file_index]}")
            self.current_file = fsspec.open(self.filepaths[self.file_index], mode="rb").open()
            self.file_index += 1
        else:
            self.current_file = None

    def read(self, size=-1):
        result = b""
        while size != 0:
            if self.current_file is None:
                break  # No more files to read from

            chunk = self.current_file.read(size)
            if not chunk:  # End of current file
                self._open_next_file()
            else:
                result += chunk
                if size > 0:
                    size -= len(chunk)
        return result

    def close(self):
        if self.current_file:
            self.current_file.close()

class JsonlPartReader(JsonlReader):
    def __init__(
            self,
            data_folder,
            adapter=None,
            text_key: str = "text",
            id_key: str = "id",
            default_metadata: dict = None,
            recursive: bool = True,
            glob_pattern: str | None = None,
    ):
        super().__init__(
            data_folder,
            adapter=adapter,
            text_key=text_key,
            id_key=id_key,
            default_metadata=default_metadata,
            recursive=recursive,
            glob_pattern=glob_pattern,
        )

    def read_files_shard(self, shard: list[str]):
        """
            Reads a list of files and yield Documents
        Args:
            shard: a list of file paths

        Returns: generator of Document

        """
        from tqdm import tqdm
        li = 0
        skipped = 0
        with (
            tqdm(
                total=self.limit if self.limit != -1 else None,
                desc="Document progress",
                unit="doc",
                disable=not self.doc_progress,
            ) as doc_pbar,
            tqdm(total=len(shard), desc="File progress", unit="file", disable=not self.file_progress) as file_pbar,
        ):
            for i, filepath in enumerate(shard):
                self.stat_update("input_files")
                di = 0
                for di, document in enumerate(self.read_file(filepath)):
                    if skipped < self.skip:
                        skipped += 1
                        continue
                    if self.limit != -1 and li >= self.limit:
                        break
                    yield document
                    doc_pbar.update()
                    li += 1
                file_pbar.update()
                self.stat_update("documents", value=di, unit="input_file")
                if self.limit != -1 and li >= self.limit:
                    break

def open_concatenated_gzip_files(filepaths):
    # Create a concatenated binary stream
    concatenated_stream = ConcatenatedFileStream(filepaths)

    # Wrap it with gzip to decompress
    gzip_stream = gzip.GzipFile(fileobj=concatenated_stream, mode='r')

    return gzip_stream

class ExtractMapccStep(PipelineStep):
    """Pipeline step to extract MAP-CC Chinese data.
    
    Reads downloaded MAP-CC files and yields documents with metadata.
    """
    def run(self, data, rank: int = 0, world_size: int = 1):
        if rank != 0:
            return
        with open_concatenated_gzip_files(data) as f:
            for li, line in enumerate(itertools.islice(f, 0, None)):
                yield orjson.loads(line)

class CollectMapccStep(PipelineStep):
    """Base pipeline block, all blocks should inherit from this one.
        Takes care of some general things such as handling dependencies, and stats

    Args:
        name: Name of the step
        type: Type of the step
            Types are high-level categories of steps, e.g. "Reader", "Tokenizer", "Filters", etc.
    """

    def run(self, data, rank: int = 0, world_size: int = 1):
        from tqdm import tqdm
        import os
        from datatrove.io import get_datafolder
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        # Initialize fsspec with Hugging Face protocol
        df = get_datafolder("hf://datasets/m-a-p/MAP-CC")
        
        # Download from m-a-p/MAP-CC to results folder
        files_to_download = []
        for file in df.list_files(recursive=True, glob_pattern="zh_cc.jsonl.gz*"):
            files_to_download.append(file)
        output_path = "/path/to/ref-datasets/mapcc"

        
        for file in files_to_download[rank::world_size]:
            print(file)
            print(f"Downloading {file}")
            output_file = f"{output_path}/{os.path.basename(file)}"
            
            # Get file size for progress bar
            file_size = df.info(file)['size']
            
            # Open input file in binary mode and write chunks directly to output
            with df.open(file, "rb") as source, open(output_file, "wb") as dest:
                with tqdm(total=file_size, unit='B', unit_scale=True, desc=os.path.basename(file)) as pbar:
                    while True:
                        chunk = source.read(8192)
                        if not chunk:
                            break
                        dest.write(chunk)
                        pbar.update(len(chunk))

if __name__ == "__main__":
    SlurmPipelineExecutor(
        job_name="mapcc-collect",
        pipeline=[
            CollectMapccStep(),
            ExtractMapccStep(),
            JsonlWriter("/path/to/ref-datasets/monolingual/zh/mapcc", output_filename="${rank}.jsonl.gz", max_file_size=2*2**30)
        ],
        logging_dir="/path/to/logs/dataset_download_logs/zh/mapcc-collect",
        randomize_start_duration=3 * 60,
        tasks=100,
        mem_per_cpu_gb=2,
        cpus_per_task=1,
        workers=10,
        partition="partition",
        time="11:59:59",
    ).run()