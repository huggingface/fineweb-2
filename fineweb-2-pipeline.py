"""
This file contains the code used to process and create the
FineWeb 2 dataset (https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)

1. we took the non english data we collected during the creation of FineWeb English as a starting point (first pipeline)
2. we then applied GlotLID for language identification covering a large number of languages
3. we filtered each language based on the language score
4. we than ran deduplication per language
5. we applied a filtering pipeline per language
6. applied some finishing touches such as pii removal, ftfy, etc
"""
from functools import partial

import yaml
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupCluster, MinhashDedupFilter, MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupBuckets
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    FineWebQualityFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
    LanguageFilter,
    URLFilter, LambdaFilter,
)
from datatrove.pipeline.formatters import PIIFormatter, FTFYFormatter, SymbolLinesFormatter
from datatrove.pipeline.readers import JsonlReader, WarcReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.hashing import HashConfig

"""
    1. The non english data from this pipeline was the starting point
"""
DUMP_TO_PROCESS = "CC-MAIN-2023-50"  # example

MAIN_OUTPUT_PATH = "s3://some_s3_bucket"
BASE_OUTPUT_PATH = f"{MAIN_OUTPUT_PATH}/base_processing"
LOGS_PATH = "/fsx/guilherme/logs/fineweb-2"

fineweb_english_executor = SlurmPipelineExecutor(
    job_name=f"cc_{DUMP_TO_PROCESS}",
    pipeline=[
        WarcReader(
            f"s3://commoncrawl/crawl-data/{DUMP_TO_PROCESS}/segments/",
            glob_pattern="*/warc/*",  # we want the warc files
            default_metadata={"dump": DUMP_TO_PROCESS},
        ),
        URLFilter(exclusion_writer=JsonlWriter(f"{BASE_OUTPUT_PATH}/removed/1_url/{DUMP_TO_PROCESS}")),
        Trafilatura(favour_precision=True),
        LanguageFilter(
            exclusion_writer=JsonlWriter(
                # THIS is the data we care about
                f"{BASE_OUTPUT_PATH}/2_non_english/{DUMP_TO_PROCESS}",
            )
        ),
        # ... other steps from FineWeb english
        JsonlWriter(f"{BASE_OUTPUT_PATH}/english/{DUMP_TO_PROCESS}"),
    ],
    tasks=8000,
    time="10:00:00",
    logging_dir=f"{LOGS_PATH}/base_processing/{DUMP_TO_PROCESS}",
    randomize_start_duration=180,  # don't hit the bucket all at once with the list requests
    mem_per_cpu_gb=2,
    partition="hopper-cpu",
)
fineweb_english_executor.run()


"""
    2. We then applied GlotLID (we actually applied it to all dumps)
"""
GLOTLID_OUTPUT_PATH = f"{MAIN_OUTPUT_PATH}/glotlid"
for dump in [
    "CC-MAIN-2023-50",
    # ...
]:
    SlurmPipelineExecutor(
        job_name=f"glotlid_{dump}",
        pipeline=[
            JsonlReader(f"{BASE_OUTPUT_PATH}/2_non_english/{DUMP_TO_PROCESS}"),
            # we keep annotations of alternative labels that are classified above 0.01
            # backend glotlid instead of ft176
            LanguageFilter(backend="glotlid", label_only=True, keep_top_pairs_threshold=0.01),
            # save in "language_script/dump"
            JsonlWriter(GLOTLID_OUTPUT_PATH,
                        output_filename="${language}_${language_script}/" + dump + "/${rank}.jsonl.gz")
        ],
        tasks=1000,
        # workers=50,
        mem_per_cpu_gb=4,
        logging_dir=f"{LOGS_PATH}/glotlid/{dump}",
        partition="hopper-cpu",
        randomize_start_duration=5 * 60,
        time="10:00:00",
    ).run()

"""
    From this point on, processing is PER LANGUAGE
"""
for lang_script in ["por_Latn", "swh_Latn", "tha_Thai", "ces_Latn"]:  #, ...]
    # we will save all data here
    LANGUAGE_OUTPUT_PATH = f"{MAIN_OUTPUT_PATH}/language/{lang_script}"

    # load the specific thresholds, stopwords etc for this language
    # we will soon share more details about how we computed them
    with open(f"configs/{lang_script}.yml") as f:
        filter_config = yaml.safe_load(f)

    def above_lang_threshold(doc, threshold):
        return doc.metadata["language_score"] >= threshold

    """
        3. Enforce a minimum language_score (which changes per language)
    """
    lang_score_filtering = SlurmPipelineExecutor(
        job_name=f"lang_filter_{lang_script}",
        pipeline=[
            JsonlReader(f"{GLOTLID_OUTPUT_PATH}/{lang_script}"),  # all the data for this language
            LambdaFilter(
                # we only keep documents with language score above the threshold
                filter_function=partial(above_lang_threshold, threshold=filter_config["language_score"]),
                exclusion_writer=JsonlWriter(
                    f"{LANGUAGE_OUTPUT_PATH}/language_filtering/removed")
            ),
            JsonlWriter(f"{LANGUAGE_OUTPUT_PATH}/language_filtering/output")
        ],
        tasks=1000,
        mem_per_cpu_gb=4,
        logging_dir=f"{LOGS_PATH}/language_filtering/{lang_script}",
        partition="hopper-cpu",
        randomize_start_duration=5 * 60,
        time="10:00:00",
    )

    """
        4. we then applied minhash deduplication to each language
    """

    # you can also change ngrams or the number of buckets and their size here
    minhash_config = MinhashConfig(
        hash_config=HashConfig(
            hash_fc="xxhash",
            precision=64,  # better precision -> fewer false positives (collisions)
        ),
        num_buckets=14,
        hashes_per_bucket=8,
        n_grams=5,
    )

    S3_MINHASH_BASE_PATH = f"{LANGUAGE_OUTPUT_PATH}/minhash"
    MINHASH_LOGS_FOLDER = f"{LOGS_PATH}/minhash/{lang_script}"

    TOTAL_TASKS = 1000

    # this is the original data that we want to deduplicate
    INPUT_READER = JsonlReader(
        f"{LANGUAGE_OUTPUT_PATH}/language_filtering/output"
    )  # this is the output from the language threshold filtering

    # stage 1 computes minhash signatures for each task (each task gets a set of files)
    stage1 = SlurmPipelineExecutor(
        job_name=f"mh1_{lang_script}",
        pipeline=[
            INPUT_READER,
            MinhashDedupSignature(
                output_folder=f"{S3_MINHASH_BASE_PATH}/signatures", config=minhash_config,
                language=lang_script # [!] THIS IS IMPORTANT: we need this to know which word tokenizer to use to split
                # into words and ngrams
            ),
        ],
        tasks=TOTAL_TASKS,
        time="5:00:00",
        partition="hopper-cpu",
        logging_dir=f"{MINHASH_LOGS_FOLDER}/signatures",
        randomize_start_duration=180,
        depends=lang_score_filtering,  # only start after the first one completes
    )

    stage2 = SlurmPipelineExecutor(
        job_name=f"mh2_{lang_script}",
        pipeline=[
            MinhashDedupBuckets(
                input_folder=f"{S3_MINHASH_BASE_PATH}/signatures",
                output_folder=f"{S3_MINHASH_BASE_PATH}/buckets",
                config=MinhashConfig(hash_config=minhash_config.hash_config),
            ),
        ],
        tasks=minhash_config.num_buckets * 50,  # the code supports parallelizing each bucket. here we run 50
        # workers per bucket
        randomize_start_duration=180,
        logging_dir=f"{MINHASH_LOGS_FOLDER}/buckets",
        partition="hopper-cpu",
        time="02:00:00",
        mem_per_cpu_gb=4,
        cpus_per_task=3,  # you can add run more (smaller) tasks if you do not have a lot of memory
        depends=stage1,
    )

    # this is the slowest step. If needed, you can use the rust version in datatrove/tools/fast_mh3
    stage3 = SlurmPipelineExecutor(
        job_name=f"mh3_{lang_script}",
        pipeline=[
            MinhashDedupCluster(
                input_folder=f"{S3_MINHASH_BASE_PATH}/buckets",
                output_folder=f"{S3_MINHASH_BASE_PATH}/remove_ids",
                config=minhash_config,
                save_cluster_size=True  # this option allows us to later upsample data based on cluster sizes
            ),
        ],
        tasks=1,  # this step runs on a single task
        logging_dir=f"{MINHASH_LOGS_FOLDER}/clusters",
        partition="hopper-cpu",
        time="30:00:00",  # and can also be quite slow. Usually not this slow though
        mem_per_cpu_gb=25,
        cpus_per_task=8,  # if you dedup a full dump, you do need a lot of memory for this one
        depends=stage2,
    )


    stage4 = SlurmPipelineExecutor(
        job_name=f"mh4_{lang_script}",
        pipeline=[
            # we must read the original input in the exact same way (nb of tasks etc)
            INPUT_READER,
            # before and after dedup
            MinhashDedupFilter(input_folder=f"{S3_MINHASH_BASE_PATH}/remove_ids"),
            JsonlWriter(f"{S3_MINHASH_BASE_PATH}/output"),
        ],
        tasks=TOTAL_TASKS,
        logging_dir=f"{MINHASH_LOGS_FOLDER}/filter",
        partition="hopper-cpu",
        time="5:00:00",
        mem_per_cpu_gb=4,
        depends=stage3,
    )

    # launch dedup pipelines
    stage4.run()


    """
        5. language specific filtering pipeline
    """
    FILTERING_OUTPUT_PATH = f"{LANGUAGE_OUTPUT_PATH}/filtering"
    SlurmPipelineExecutor(
        job_name=f"filter_{lang_script}",
        pipeline=[
            # read minhashed data
            JsonlReader(f"{S3_MINHASH_BASE_PATH}/output"),
            # gopher repetition filter
            GopherRepetitionFilter(
                language=lang_script,  # [!] THIS IS IMPORTANT: we need this to know which word tokenizer to use to split
                # into words and ngrams
                # we disable these. trafilatura pretty much removes paragraph and we use a different threshold
                # for dup_line_char_frac in fineweb quality
                dup_para_frac=0,
                dup_line_char_frac=0,
                dup_para_char_frac=0,
                dup_line_frac=filter_config['dup_line_frac'],
                top_n_grams=filter_config["top_n_grams"],
                dup_n_grams=filter_config["dup_n_grams"],
                exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/goph_rep/")
            ),
            # fineweb quality
            FineWebQualityFilter(
                language=lang_script,
                short_line_thr=999,  # we disable this filter
                char_duplicates_ratio=0.1,  # we changed this from 0.01 in fineweb english
                line_punct_thr=filter_config["line_punct_thr"],
                new_line_ratio=filter_config['new_line_ratio'],
                exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/fw_qual/")
            ),
            # gopher quality filter
            GopherQualityFilter(
                language=lang_script,
                max_avg_word_length=filter_config['max_avg_word_length'],
                min_avg_word_length=filter_config['min_avg_word_length'],
                stop_words=filter_config['stopwords'],
                max_non_alpha_words_ratio=filter_config['max_non_alpha_words_ratio'],
                min_stop_words=2,
                exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/goph_qual/")
            ),
            # we do not apply the C4 filters
            JsonlWriter(f"{FILTERING_OUTPUT_PATH}/output"),
        ],
        tasks=TOTAL_TASKS,
        logging_dir=f"{LOGS_PATH}/filtering/{lang_script}",
        partition="hopper-cpu",
        time="5:00:00",
        mem_per_cpu_gb=4,
        depends=stage3,
    ).run()

    """
        6. final touches
    """
    SlurmPipelineExecutor(
        job_name=f"final_touches_{lang_script}",
        pipeline=[
            JsonlReader(f"{FILTERING_OUTPUT_PATH}/output"),
            FTFYFormatter(),  # fix encoding issues. Important in a multilingual setting
            PIIFormatter(),  # remove PII
            SymbolLinesFormatter(symbols_to_remove=["|"], replace_char="\n"),  # fix trafilatura table artifacts
            JsonlWriter(f"{LANGUAGE_OUTPUT_PATH}/final_touches"),
        ],
        tasks=TOTAL_TASKS,
        logging_dir=f"{LOGS_PATH}/final_touches/{lang_script}",
        partition="hopper-cpu",
        time="5:00:00",
        mem_per_cpu_gb=4,
    ).run()