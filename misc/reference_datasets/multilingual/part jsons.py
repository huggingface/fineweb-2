from tqdm import tqdm
import itertools
from orjson import orjson
import fsspec
import gzip
from io import BytesIO


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


def open_concatenated_gzip_files(filepaths):
    # Create a concatenated binary stream
    concatenated_stream = ConcatenatedFileStream(filepaths)

    # Wrap it with gzip to decompress
    gzip_stream = gzip.GzipFile(fileobj=concatenated_stream, mode='r')

    return gzip_stream


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
                logger.info(f"Reading input file {filepath}, {i + 1}/{len(shard)}")
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

    def read_file(self, filepath: str):
        import orjson
        from orjson import JSONDecodeError

        with self.data_folder.open(filepath, "r", compression=self.compression) as f:
            try:
                for li, line in enumerate(f):
                    with self.track_time():
                        try:
                            document = self.get_document_from_dict(orjson.loads(line), filepath, li)
                            if not document:
                                continue
                        except (EOFError, JSONDecodeError) as e:
                            logger.warning(f"Error when reading `{filepath}`: {e}")
                            continue
                    yield document
            except UnicodeDecodeError as e:
                logger.warning(f"File `{filepath}` may be corrupted: raised UnicodeDecodeError ({e})")


# Example usage:
filepaths = ["zh_baike.jsonl.gz.partaa", "zh_baike.jsonl.gz.partab"]
with open_concatenated_gzip_files(filepaths) as f:
    for li, line in enumerate(itertools.islice(f, 0, None)):
        # have a party
        # Process each line
        if li % 100000 == 0:
            print(orjson.loads(line))  # Assuming the content is UTF-8 encoded