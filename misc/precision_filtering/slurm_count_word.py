import pandas as pd

# Load the dataset
url = 'https://raw.githubusercontent.com/huggingface/datatrove/main/src/datatrove/assets/tokenizer_assignment.csv'
df = pd.read_csv(url)


# Function to generate a tokenizer identifier
def generate_tokenizer_identifier(row):
    return f"{row['type']}-{row['tok_code']}"

df = df.dropna(subset=['type'])
# df = df[df['type'] != 'StanzaTokenizer']

# Apply function to create a new column
df['tokenizer_identifier'] = df.apply(generate_tokenizer_identifier, axis=1)
df['lang_identifier'] = df.apply(lambda x: f"{x['code_3']}_{x['script']}", axis=1)

# Group by tokenizer and map to language codes
tokenizer_to_example_language = df.groupby('tokenizer_identifier')['lang_identifier'].first().to_dict()

# Limit
tokenizer_to_example_language = {k: tokenizer_to_example_language[k] for k in ['BurmeseTokenizer-nan', 'TibetanTokenizer-nan', 'SpaCyTokenizer-yo'] if k in tokenizer_to_example_language}

from datatrove.pipeline.base import PipelineStep
class ComputeTermFrequency(PipelineStep):
    def __init__(self, tokenizer_name: str, example_language: str):
        self.tokenizer_name = tokenizer_name
        self.example_language = example_language

    def run(self, _data, rank, world_size):
        import os
        from datatrove.utils.text import split_into_words, simplify_text, TextNormConfig
        from collections import Counter
        import pickle

        norm_config = TextNormConfig(
            lowercase=False,
            norm_numbers=False,
            norm_weekdays=False,
            norm_monthnames=False,
            remove_punctuation=True,
            norm_unicode_diacritics=False,
            norm_whitespace=True,
        )
        root_dir = "/fsx/user_dir/glotlid-corpus/v3.1/"  # Change this to your actual directory from glotlid corpus or another corpus
        # List of language codes
        language_codes = [x for x in os.listdir(root_dir)]
        language_codes.sort()
        # shard that we will process
        language_codes = language_codes[rank::world_size]

        base_save_path = f"/fsx/user_dir/language_tf/{self.tokenizer_name}" # the save path for the term frequencies
        os.makedirs(base_save_path, exist_ok=True)

        for code in language_codes:
            dir_path = os.path.join(root_dir, f"{code}")
            print(f"Processing language code {code} with tokenizer {self.tokenizer_name} ({self.example_language})")

            if os.path.exists(os.path.join(base_save_path, f"{code}.pkl")):
                continue

            word_counts = Counter()

            if os.path.isdir(dir_path):
                print(f"Processing {dir_path}...")

                # Iterate over all text files in the directory
                for filename in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, filename)
                    
                    if filename.endswith(".txt") and os.path.isfile(file_path):
                        with open(file_path, "r", encoding="utf-8") as f:
                            # Read up to 1000 lines and add to the list
                            for line in f:
                                # removed the 100k limit, feel free to re-add if you thing it will be an issue
                                simplified = simplify_text(line.strip(), norm_config)
                                # use example_language as an example for this tokenizer
                                word_counts.update(split_into_words(simplified, self.example_language))
            with open(os.path.join(base_save_path, f"{code}.pkl"), "wb") as f:
                pickle.dump(word_counts, f)

from datatrove.executor.slurm import SlurmPipelineExecutor
for tokenizer_name, example_language in tokenizer_to_example_language.items():
    executor = SlurmPipelineExecutor(
        pipeline=[
            ComputeTermFrequency(tokenizer_name, example_language)
        ],
        tasks=100, # increase as needed. basically workers per tokenizer
        partition="hopper-cpu",
        qos="normal",
        time="12:00:00",
        cpus_per_task=2,
        sbatch_args={
            "mem-per-cpu": "1950M"  # small optimization
        },
        env_command="source ~/.bashrc && conda activate myenv", #myenv is the conda environment with datatrove installed
        logging_dir=f"/fsx/user_dir/language_tf-2-logs/{tokenizer_name}", # logging directory for each tokenizer
        job_name=f"glot-{tokenizer_name}"
    )
    executor.run()