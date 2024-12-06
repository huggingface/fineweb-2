import argparse

parser = argparse.ArgumentParser("Quickly launch thom's style of tokenization.")

parser.add_argument(
    "data_path", type=str, help="Path to the data to tokenize."
)
parser.add_argument(
    "output_name", type=str, help="Output name."
)
parser.add_argument(
    "--n_tasks", type=int, help="nb of tokenization tasks", default=1000
)
parser.add_argument(
    "--max_toks", type=int, help="max tokens per file", default=1e8
)
parser.add_argument(
    "--tokenizer", type=str, help="tokenizer to use", default="google/gemma-2b"
)
parser.add_argument(
    "--text_key", type=str, default="text"
)
parser.add_argument(
    "--sample", type=float, default=1.0
)
parser.add_argument("--qos", type=str, default="normal", help="qos to use")
parser.add_argument(
    "--jsonl_output", "-jo", type=str, default=None, help="Path to optionally save the sampled data jsonl"
)
parser.add_argument("-d", help="dependency job", type=str, default=None)
if __name__ == "__main__":
    args = parser.parse_args()
    from datatrove.executor import SlurmPipelineExecutor
    from datatrove.pipeline.filters import SamplerFilter
    from datatrove.pipeline.readers import JsonlReader
    from datatrove.pipeline.writers import JsonlWriter
    from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer

    SlurmPipelineExecutor(
        # job_name=f"nd-{DUMP_NUMBER}-{len(DUMPS)}",
        job_name=f"tok-{args.output_name}",
        pipeline=[
            JsonlReader(
                args.data_path,
                text_key=args.text_key,
            ),
            SamplerFilter(rate=args.sample),
            *([JsonlWriter(args.jsonl_output)] if args.jsonl_output else []),
            DocumentTokenizer(
                output_folder=f"/path/to/tokenized/{args.output_name}",
                local_working_dir=f"/scratch/$USER/multilingual/tok/{args.output_name}",
                tokenizer_name_or_path=args.tokenizer,
                eos_token=None,
                batch_size=10000,
                max_tokens_per_file=args.max_toks,
                # Max 1 GT per file (i.e. btw 5 et 300 tokenized files per dump et about 100 dump extracts per merged file)
                shuffle=True,
            ),
        ],
        tasks=args.n_tasks,
        time="2:00:00",
        partition="hopper-cpu",
        logging_dir=f"/path/to/logs/multilingual/toks/{args.output_name}",
        cpus_per_task=32,
        qos=args.qos,
        mem_per_cpu_gb=3,
        depends_job_id=args.d,
    ).run()