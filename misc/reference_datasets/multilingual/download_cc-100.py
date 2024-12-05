from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep


class CC100Reader(PipelineStep):
    def run(self, data=None, rank: int = 0, world_size: int = 1):
        languages = [
            "af",
            "am",
            "ar",
            "as",
            "az",
            "be",
            "bg",
            "bn",
            "bn_rom",
            "br",
            "bs",
            "ca",
            "cs",
            "cy",
            "da",
            "de",
            "el",
            "en",
            "eo",
            "es",
            "et",
            "eu",
            "fa",
            "ff",
            "fi",
            "fr",
            "fy",
            "ga",
            "gd",
            "gl",
            "gn",
            "gu",
            "ha",
            "he",
            "hi",
            "hi_rom",
            "hr",
            "ht",
            "hu",
            "hy",
            "id",
            "ig",
            "is",
            "it",
            "ja",
            "jv",
            "ka",
            "kk",
            "km",
            "kn",
            "ko",
            "ku",
            "ky",
            "la",
            "lg",
            "li",
            "ln",
            "lo",
            "lt",
            "lv",
            "mg",
            "mk",
            "ml",
            "mn",
            "mr",
            "ms",
            "my",
            "my_zaw",
            "ne",
            "nl",
            "no",
            "ns",
            "om",
            "or",
            "pa",
            "pl",
            "ps",
            "pt",
            "qu",
            "rm",
            "ro",
            "ru",
            "sa",
            "si",
            "sc",
            "sd",
            "sk",
            "sl",
            "so",
            "sq",
            "sr",
            "ss",
            "su",
            "sv",
            "sw",
            "ta",
            "ta_rom",
            "te",
            "te_rom",
            "th",
            "tl",
            "tn",
            "tr",
            "ug",
            "uk",
            "ur",
            "ur_rom",
            "uz",
            "vi",
            "wo",
            "xh",
            "yi",
            "yo",
            "zh-Hans",
            "zh-Hant",
            "zu"
        ]
        from fsspec import open as fsspec_open
        def get_doc_texts(file):
            with fsspec_open(file, mode="rt", compression="xz") as f:
                lines = []
                for line in f:
                    if line == "\n":
                        yield "".join(lines).strip()
                        lines = []
                    lines.append(line)
                if lines:
                    yield "".join(lines).strip()

        from loguru import logger
        from datatrove.data import Document
        if rank >= len(languages):
            return
        from datatrove.pipeline.writers import JsonlWriter
        lang = languages[rank]
        logger.info(f"Processing \"{lang}\"")

        with JsonlWriter(f"/path/to/ref-datasets/cc-100/{lang.lower()}", max_file_size=200 * 2**20) as writer:
            for doci, doctext in enumerate(get_doc_texts(f"/path/to/data/cc-100/{lang}.txt.xz")):
                doc = Document(
                    text=doctext,
                    id=f"cc-100/{lang}/{doci}",
                    metadata={
                        "lang": lang
                    }
                )
                writer.write(doc)


SlurmPipelineExecutor(
    job_name="cc100",
    pipeline=[
        CC100Reader(),
    ],
    tasks=120,
    mem_per_cpu_gb=4,
    cpus_per_task=4,
    logging_dir="/path/to/logs/multilingual/copy/cc-100",
    partition="hopper-cpu",
    time="20:00:00"
).run()
