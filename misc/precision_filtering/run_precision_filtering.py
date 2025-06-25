
import json
from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from functools import lru_cache
from datatrove.pipeline.filters.base_filter import BaseFilter


# filter used for wordlist/url filtering
class Decontaminate(BaseFilter):
    def __init__(self, lang_code, language, 
        whitelist_words: list[str] = [], exclusion_writer = None):
        super().__init__(exclusion_writer)
        from datatrove.utils.text import TextNormConfig
        self.language = language
        self.norm_config = TextNormConfig(
            lowercase=False,
            norm_numbers=False,
            norm_weekdays=False,
            norm_monthnames=False,
            remove_punctuation=True,
            norm_unicode_diacritics=False,
            norm_whitespace=True,
        )

        # url related filtering
        self.lang_code = lang_code
        self.whitelist_words = whitelist_words
        import ahocorasick
        from tldextract import TLDExtract
        self.tldextractor = TLDExtract()
        self.domain_extension = None
        domain_extension = [x for x in self.whitelist_words if x.startswith(".")]
        if domain_extension:
            self.domain_extension = domain_extension[0] + "/"

        self.whitelist_words = [x for x in self.whitelist_words if not x.startswith(".")]
        
        self.whitelist_automaton = ahocorasick.Automaton(ahocorasick.STORE_INTS)
        for word in self.whitelist_words:    
            self.whitelist_automaton.add_word(word, len(self.whitelist_automaton))
        self.whitelist_automaton.make_automaton()

        import re
        self.normalizer = re.compile(r"[^a-zA-Z0-9\/\.]+")  # we include / and . to separate url sections
        self.lang_code_pattern = re.compile(rf'(?<![a-zA-Z0-9]){self.lang_code}(?![a-zA-Z0-9])')

    @lru_cache(maxsize=1)
    def wordlist(self):
        import os
        wordlist_path = os.path.join("wordlists-0.85", f"{self.language}.txt")
        if not os.path.exists(wordlist_path):
            raise ValueError(f"Wordlist for {self.language} not found")
        return set(line.strip() for line in open(wordlist_path, "r") if line.strip())

    def wordlist_filter(self, doc):
        from datatrove.utils.text import split_into_words, simplify_text
        text = simplify_text(doc.text, self.norm_config)
        words = set(split_into_words(text, self.language))
        matching_words = len(words & self.wordlist())
        doc.metadata["wordlist_ratio"] = matching_words / len(words) if words else 0
        return doc.metadata["wordlist_ratio"] > 0.0000001

    def filter(self, document):
        return self.url_filter(document) or self.wordlist_filter(document)

    def url_filter(self, document):
        url = document.metadata.get("url").removesuffix("http://").removesuffix("https://")

        assert url, "Document does not have url in its metadata"
        url_info = self.tldextractor(url)

        # check domain extension
        if self.domain_extension and self.domain_extension in url_info.fqdn:
            document.metadata['url_match'] = self.domain_extension
            # print(f"DOMAIN EXTENSION: {self.domain_extension} in {url_info.fqdn}")
            return True

        # check lang code (pre space normalization)
        if self.lang_code.upper() in url or self.lang_code_pattern.search(url):
            # document.metadata['url_match'] = self.lang_code.upper()
            # print(f"LANG CODE: {self.lang_code} in {url}")
            return True
        
        # check whitelist words
        normalized_space = self.normalizer.sub("", url).lower()
        if not self.whitelist_words:
            return False
        found = list(self.whitelist_automaton.iter(normalized_space))
        if found:
            words = [
                self.whitelist_words[value] for end_index, value in found
            ]
            # print(f"WHITELIST WORDS: {words} in {url}")
            # document.metadata['url_match'] = words
            return True

        return False

with open("url_whitelist.json", "rt") as f:
    url_whitelist_data = json.load(f)

with open("cont_scores.json", "rt") as f:
    cont_scores = json.load(f)

for language, cont_score in cont_scores.items():
    if cont_score <= 0.1:
        print(f"Skipping {language}: contamination score is too low, not running precision filtering")
        continue

    lang_code = language.split("_")[0]
    url_whitelist_words = url_whitelist_data.get(lang_code, [])
    execut = SlurmPipelineExecutor(
        pipeline=[
            JsonlReader(
                f"s3://data_to_filter/output/{language}"
            ),
            Decontaminate(lang_code, language, whitelist_words=url_whitelist_words, exclusion_writer=JsonlWriter(f's3://data_to_filter/precision_filtering/removed/{language}')),
            JsonlWriter(
                f"s3://data_to_filter/precision_filtering/output/{language}"
            ),
        ],
        tasks=10,
        time="30:00:00",
        partition="hopper-cpu",
        cpus_per_task=1,
        qos="normal",
        sbatch_args={
            "mem-per-cpu": "1950M",
        },
        env_command="sleep $((RANDOM % 30))",
        logging_dir=f"logs/precision_filtering/{language}",
        job_name=f"{language}_precf",
    )
    
    execut.run()