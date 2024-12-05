[TODO ICON HERE]
# Fineweb 2

Fineweb 2 is continuation of our Fineweb dataset, expanding the langauge support from English to 1862 languages!

## Data Pipeline
Compared to the original Fineweb, the data pipeline got a lot of updates. As each langauge has its own peculiarities, configuration for each language differ, you can view the configuration for each language in `/configs/{iso3_lang}_{script}.yml`.

### Language Identification
- Done using [GlotLID](https://github.com/cisnlp/GlotLID), which not only covers way wider variety of languages, but also support disction on script level. For each language we decided on threshold follwing formula:
    - `<INSERT FORMULA HERE>`
- Threshold for each language is `<INSERT VALUE HERE>` [TODO: this should be upload]

### Data Deduplication
- Deduplication is now done done before filtering and globally (not per CommonCrawl shard), for each language correct word tokenizer is chosen to achieve best deduplication results.

### Data Filtering
- Most of the filters are the same as in the original Fineweb, with some updates:
    - Removed `short_line_filter` as well as changing `char_dup_ratio` from 0.01 to 0.1.
    - Stopwords are automatically infered from wikipedia for each language.
    - Similarly thresholds for `line_punct_ratio` and `<and other please insert which ones>` are language specific.

### PII Anonymization
- Kept unchanged, emails and ip addresses are anonymized.

## Ablations code
Each design choice in the pipeline was carefully tested on our created evaluation suite FineTasks <reference>. After each pipeline step, we trained a 1.XXXB ?? model for each language using a nanotron framework and then evaluated one of the tasks. For full transparency we provide both evaluation and training code.

- [Evaluation code](https://github.com/HuggingFaceFW/fineweb-2/tree/main/evaluations)
- [Training code](https://github.com/HuggingFaceFW/fineweb-2/tree/main/modelling)
- [Tokenization code](https://github.com/HuggingFaceFW/fineweb-2/tree/main/tokenization)

## Misc

### Word/Sentence Tokenization
Many parts of the pipeline rely on correct word and sentence tokenization, however for most of the languages there are no such tokenizers available. We have thus developed a method to automatically assign similar tokenizer based on language family. You can check the tokenizer for each language at `<Insert link>`.

### Reference Datasets
For full transparency, we provide code used to download and process reference datasets in `/misc/reference_datasets`.

### Tools version
- [Datatrove](https://github.com/HuggingFaceFW/datatrove) - <Add commit>
- [LightEval](https://github.com/HuggingFaceFW/ml-lighteval) - <Add commit>
- [Nanotron](https://github.com/HuggingFaceFW/nanotron) - <Add commit>

## License
The dataset retains the same license as the original Fineweb, which is Open Data Commons License Attribution family (ODC-By).
