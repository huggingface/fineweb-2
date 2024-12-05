from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter


for lang in ('tr', 'ar', 'zh'):
    SlurmPipelineExecutor(
        job_name=f"hplt_per_{lang}",
        pipeline=[
            JsonlReader(f"s3://path/ref-datasets/hplt-mono/{lang}"),
            JsonlWriter(f"s3://path/ref-datasets/hplt-mono-split/{lang}", max_file_size=2**30)
        ],
        tasks=200,
        mem_per_cpu_gb=4,
        # workers=5,
        logging_dir=f"/path/to/logs/multilingual/copy/hplt-mono-split/{lang}",
        partition="hopper-cpu",
        cpus_per_task=1,
        randomize_start_duration=60,
        qos="high",
        time="20:00:00",
        sbatch_args={"constraint": "cpu"}
    ).run()
