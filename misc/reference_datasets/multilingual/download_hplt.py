from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.readers import ParquetReader, JsonlReader
from datatrove.pipeline.writers import JsonlWriter


class HPLTReader(JsonlReader):

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
        import random
        if data:
            yield from data
        with self.data_folder.open("hplt_monolingual_map_cleaned_1.2.txt", "rt") as f:
            files = [path.removeprefix("https://data.hplt-project.org/one/monotext/cleaned/") for path in
                     f.read().splitlines()]
        files_shard = files[rank::world_size]
        if len(files_shard) == 0:
            if rank == 0:
                raise RuntimeError(f"No files found on {self.data_folder.path}!")
            # otherwise just a warning
            logger.warning(f"No files found on {self.data_folder.path} for {rank=}")
        if self.shuffle_files:
            random.shuffle(files_shard)
        for doc in self.read_files_shard(files_shard):
            self.update_doc_stats(doc)
            yield doc

SlurmPipelineExecutor(
    job_name="hplt",
    pipeline=[
        HPLTReader("https://data.hplt-project.org/one/monotext/cleaned"),
        JsonlWriter("/path/to/ref-datasets/hplt-mono",
                    output_filename="${document_lang}" + "/${rank}.jsonl.gz")
    ],
    tasks=1000,
    mem_per_cpu_gb=4,
    workers=5,
    logging_dir="/path/to/logs/multilingual/copy/hplt-mono",
    partition="partition",
    cpus_per_task=2,
    sbatch_args={
        "constraint": "cpu"
    },
    randomize_start_duration=3 * 60,
    time="20:00:00"
).run()
