import pandas as pd
import os
import pickle
from collections import Counter

# Load the dataset
url = 'https://raw.githubusercontent.com/huggingface/datatrove/main/src/datatrove/assets/tokenizer_assignment.csv'
df = pd.read_csv(url)

# Function to generate a tokenizer identifier
def generate_tokenizer_identifier(row):
    return f"{row['type']}-{row['tok_code']}"

df = df.dropna(subset=['type'])

# Apply function to create a new column
df['tokenizer_identifier'] = df.apply(generate_tokenizer_identifier, axis=1)
df['lang_identifier'] = df.apply(lambda x: f"{x['code_3']}_{x['script']}", axis=1)


# Group by tokenizer and map to language codes
tokenizer_to_languages = df.groupby('tokenizer_identifier')['lang_identifier'].apply(set).apply(list).to_dict()


root_path = "/fsx/user_dir/language_tf/"
save_path = "/fsx/user_dir/common_freq/"
corpus_path = '/fsx/user_dir/glotlid-corpus/v3.1/'
corpus_size = 1 * len(os.listdir(corpus_path)) # Adjust corpus_size as needed

languages = {}
for tokenizer_id in tokenizer_to_languages.keys():
    tokenizer_path = os.path.join(root_path, tokenizer_id)

    if len(os.listdir(tokenizer_path))  >= corpus_size:

        for language in tokenizer_to_languages[tokenizer_id]:

            if os.path.exists(os.path.join(save_path, language + '.pkl')):
                continue
            if os.path.exists(os.path.join(corpus_path, language)):
                languages[language] = tokenizer_id


def load_and_save_tokenizer_freq(tokenizer_id, selected_language):
    # Load selected langauge
    tokenizer_path = os.path.join(root_path, tokenizer_id)
    with open(os.path.join(tokenizer_path, selected_language + '.pkl'), 'rb') as f:
        language_tf = pickle.load(f)

    os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists

    common_freq = Counter()

    # Iterate over all pickle files in the directory
    for filename in os.listdir(tokenizer_path):
        if filename.endswith(".pkl"):
            file_path = os.path.join(tokenizer_path, filename)
            
            with open(file_path, "rb") as f:
                counts = pickle.load(f)
                # Filter counts based on the keys present in language_tf (which is a Counter)
                filtered_counts = {word: counts[word] for word in language_tf if word in counts}
                common_freq.update(filtered_counts)  # Only update with relevant words

    # Save the common_freq Counter to a pickle file
    save_path = os.path.join(save_path, f"{selected_language}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(common_freq, f)


# Iterate over the languages and load/save the tokenizer frequencies - You can do it in parallel if needed
for selected_language, tokenizer_id in languages.items():
    print(selected_language, tokenizer_id)
    load_and_save_tokenizer_freq(tokenizer_id, selected_language)
