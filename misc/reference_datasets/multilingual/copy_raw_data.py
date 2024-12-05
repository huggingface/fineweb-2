from time import sleep

from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
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
        "text": data.pop(self.text_key, data.pop("content", "")),
        "id": data.pop(self.id_key, data.pop("data-id", f"{path}/{id_in_file}")),
        "media": data.pop("media", []),
        "metadata": data.pop("metadata", {}) | data,
        # remaining data goes into metadata
    }


class CachedListReader(JsonlReader):
    def __init__(self,
                 data_folder,
                 dump_to_proc: str,
                 compression="infer",
                 limit: int = -1,
                 skip: int = 0,
                 file_progress: bool = False,
                 doc_progress: bool = False,
                adapter = None,
                 text_key: str = "text",
                 id_key: str = "id",
                 default_metadata: dict = None,
                 recursive: bool = True,
                 glob_pattern: str | None = None,
                 shuffle_files: bool = False):
        super().__init__(
            data_folder,
            None,
            compression,
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
        self.dump_to_proc = dump_to_proc

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        """
        Will get this rank's shard and sequentially read each file in the shard, yielding Document.
        Args:
            data: any existing data from previous pipeline stages
            rank: rank of the current task
            world_size: total number of tasks

        Returns:

        """
        from loguru import logger
        if data:
            yield from data
        files_shard = []
        pathi = 0
        with open("/path/to/base_proc_filelist.txt", "rt") as f:
            for path in f:
                if path.split("/")[1] == self.dump_to_proc:
                    if (pathi - rank) % world_size == 0:
                        files_shard.append(path.strip())
                    pathi += 1
        logger.info(f"Loaded {len(files_shard)} for {rank=}, {world_size=}")
        if len(files_shard) == 0:
            if rank == 0:
                raise RuntimeError(f"No files found on {self.data_folder.path}!")
            # otherwise just a warning
            logger.warning(f"No files found on {self.data_folder.path} for {rank=}")
        for doc in self.read_files_shard(files_shard):
            self.update_doc_stats(doc)
            yield doc


    def read_file(self, filepath: str):
        import orjson
        from orjson import JSONDecodeError
        from loguru import logger

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
            except Exception as e:
                if "Error -3 while decompressing data" in str(e):
                    logger.warning(f"CORRUPTED `{filepath}`: {e}")
                else:
                    logger.warning(f"Unknwon: {e}")


for dump in """CC-MAIN-2014-42
CC-MAIN-2014-23
CC-MAIN-2014-41
CC-MAIN-2014-35
CC-MAIN-2014-15
CC-MAIN-2014-10
CC-MAIN-2013-48
CC-MAIN-2017-17
CC-MAIN-2017-13
CC-MAIN-2017-09
CC-MAIN-2015-18
CC-MAIN-2016-44
CC-MAIN-2014-52
CC-MAIN-2014-49
CC-MAIN-2015-22
CC-MAIN-2022-21
CC-MAIN-2023-23
CC-MAIN-2022-49
CC-MAIN-2017-04
CC-MAIN-2023-40
CC-MAIN-2023-14
CC-MAIN-2023-50
CC-MAIN-2021-43
CC-MAIN-2015-35
CC-MAIN-2016-50
CC-MAIN-2013-20
CC-MAIN-2015-48
CC-MAIN-2023-06
CC-MAIN-2015-11
CC-MAIN-2022-40
CC-MAIN-2015-32
CC-MAIN-2015-06
CC-MAIN-2021-31
CC-MAIN-2022-27
CC-MAIN-2021-04
CC-MAIN-2016-07
CC-MAIN-2017-43
CC-MAIN-2020-40
CC-MAIN-2016-30
CC-MAIN-2021-17
CC-MAIN-2015-27
CC-MAIN-2016-40
CC-MAIN-2021-39
CC-MAIN-2015-14
CC-MAIN-2022-05
CC-MAIN-2020-05
CC-MAIN-2017-34
CC-MAIN-2020-29
CC-MAIN-2017-26
CC-MAIN-2018-05
CC-MAIN-2018-09
CC-MAIN-2016-36
CC-MAIN-2017-22
CC-MAIN-2018-30
CC-MAIN-2020-16
CC-MAIN-2017-47
CC-MAIN-2018-51
CC-MAIN-2017-30
CC-MAIN-2019-35
CC-MAIN-2018-13
CC-MAIN-2019-43
CC-MAIN-2021-10
CC-MAIN-2017-39
CC-MAIN-2021-21
CC-MAIN-2022-33
CC-MAIN-2018-26
CC-MAIN-2020-45
CC-MAIN-2017-51
CC-MAIN-2019-09
CC-MAIN-2016-22
CC-MAIN-2021-49
CC-MAIN-2018-43
CC-MAIN-2018-17
CC-MAIN-2020-50
CC-MAIN-2021-25
CC-MAIN-2015-40
CC-MAIN-2020-24
CC-MAIN-2019-47
CC-MAIN-2024-10
CC-MAIN-2019-22
CC-MAIN-2019-04
CC-MAIN-2016-18
CC-MAIN-2019-30
CC-MAIN-2018-47
CC-MAIN-2019-39
CC-MAIN-2018-39
CC-MAIN-2019-18
CC-MAIN-2019-26
CC-MAIN-2020-34
CC-MAIN-2020-10
CC-MAIN-2019-51
CC-MAIN-2019-13
CC-MAIN-2018-22
CC-MAIN-2018-34
CC-MAIN-2024-18
CC-MAIN-2016-26""".splitlines():
    SlurmPipelineExecutor(
        job_name=f"cp_{dump}",
        pipeline=[
            CachedListReader("/path/to/base_processing/non_english",
                             adapter=adapter, dump_to_proc=dump),
            JsonlWriter(f"/path/to/raw_fw/{dump}")
        ],
        tasks=5000,
        mem_per_cpu_gb=4,
        cpus_per_task=2,
        logging_dir=f"/path/to/logs/multilingual/copy/raw_fw/{dump}",
        partition="partition",
        randomize_start_duration=3 * 60,
        time="20:00:00",
        max_array_launch_parallel=True
    ).run()
