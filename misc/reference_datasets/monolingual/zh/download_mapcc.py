import fsspec
from datatrove.executor import LocalPipelineExecutor, SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep

class ExtractMapccStep(PipelineStep):
    """Pipeline step to extract MAP-CC Chinese data.
    
    Reads downloaded MAP-CC files and yields documents with metadata.
    """
    def run(self, data, rank: int = 0, world_size: int = 1):
        if rank != 0:
            return
        import gzip
        import json
        import os
        from tqdm import tqdm
        from datatrove.io import get_datafolder
        # Get list of downloaded files
        input_path = "/fsx/hynek_kydlicek/datasets/mapcc"
        df = get_datafolder(input_path)
        files = df.list_files(recursive=True, glob_pattern="zh_cc.jsonl.gz*")

        # Process each gzipped file
        for filename in tqdm(files, desc="Processing files"):
            input_file = os.path.join(input_path, filename)
            
            # Read gzipped jsonl file
            with gzip.open(input_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        doc = json.loads(line)
                        # Add language metadata
                        doc['metadata'] = {'language': 'zh'}
                        yield doc
                    except json.JSONDecodeError:
                        continue

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
            CollectMapccStep()
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