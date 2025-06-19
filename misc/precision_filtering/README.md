# Running wordlist precision filtering

Wordlists are provided in the `wordlists-0.85` and `wordlists-0.95` directories, for thresholds of 85% and 95%, respectively (see 3. below for more info).
Since some wordlists may be too strict, we additionally keep documents whose URL contains terms related to each language (the actual language code, the name of the language, possibly top level domains for that region and/or names of regions where the language is spoken). These are saved in [url_whitelist.json](url_whitelist.json).

To run the filtering, refer to the following `datatrove` script:
- [run_precision_filtering.py](run_precision_filtering.py) filter languages that had a contamination score above 10% (from our data) using the wordlists in `wordlists-0.85` and `url_whitelist.json`

# Generating wordlists for precision filtering
This directory also contains scripts for generating high-precision wordlists for contamination detection. The pipeline consists of four main steps:

## 1. Compute Term Frequencies (slurm_count_word.py)
Run `slurm_count_word.py` first. This script calculates the appearance (term frequency) of all words in each language for all available tokenizers. It uses the GlotLID corpus and saves the word frequency counts for each language-tokenizer pair as pickle files.

**How to run:**
- Make sure you have the required environment and dependencies (see script for details).
- Submit the script to your SLURM cluster:
  ```bash
  python slurm_count_word.py
  ```

## 2. Count Common Words (count_common.py)
Next, run `count_common.py`. This script selects the correct tokenizer for each language and computes, for each language, how many of its words also appear in other languages (using the same tokenizer). The results are saved as pickle files in the `common_freq` directory.

**How to run:**
```bash
python count_common.py
```

## 3. Generate High-Precision Wordlists (wordlist_gen.py)
Then, run `wordlist_gen.py`. This script selects only those words that appear at least 95% (can be changed) of the time in a given language (by dividing the language-specific term frequencies from step 1 by the common word counts from step 2). The output is a set of wordlists (one per language) saved in the `wordlists-0.95` directory. We additionally provide `wordlists-0.85`, with words selected using a threshold of 85%, which is the one that was ultimately used in the creation of FineWeb2.

**How to run:**
```bash
python wordlist_gen.py
```

## 4. Score Texts for Contamination (wordlist_score.py)
Finally, use `wordlist_score.py` to decide if a text is contaminated or not. This script loads the generated wordlists and provides a function to score a text based on the proportion of its words that appear in the high-precision wordlist for a given language.

**How to use:**
- Import or run the script and use the `filter_score(text, lang)` function to get a wordlist score for your text.
- Example usage is provided at the end of the script.

---

**Summary:**
1. `slurm_count_word.py`: Compute word frequencies for all languages and tokenizers.
2. `count_common.py`: For each language, count how many of its words also appear in other languages (with the same tokenizer).
3. `wordlist_gen.py`: Select words that are highly specific to each language (≥95% precision).
4. `wordlist_score.py`: Use the generated wordlists to score and detect contamination in new texts.

See each script for more details and customization options.

