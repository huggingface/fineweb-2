import pandas as pd
import os
import pickle
from collections import Counter
import numpy as np

def filter_top_percentile(counter, percentile=95):
    # Get the frequencies as a list
    frequencies = list(counter.values())
    
    # Calculate the percentile threshold
    percentile_threshold = np.percentile(frequencies, percentile)
    
    # Filter and sort the counter by descending count
    filtered_items = {
        word: count for word, count in counter.items() if count >= percentile_threshold
    }
    sorted_filtered = dict(sorted(filtered_items.items(), key=lambda x: x[1], reverse=True))
    
    return Counter(sorted_filtered)



def filter_by_ratio(counter1, counter2, threshold=0.95):
    # Create a new Counter to store the results
    filtered_counter = Counter()

    # Iterate over the keys of the first counter
    for key, value1 in counter1.items():
        # Check if the key exists in the second counter
        if key in counter2:
            value2 = counter2[key]
            # Compute the ratio and check if it's above the threshold
            ratio = value1 / value2
            if ratio >= threshold:
                filtered_counter[key] = ratio
    
    return filtered_counter



def save(tokenizer_id, selected_language):

    tokenizer_path = os.path.join(root_path, tokenizer_id)

    with open(os.path.join(tokenizer_path, selected_language + '.pkl'), 'rb') as f:
        language_tf = pickle.load(f)

    with open(os.path.join('/fsx/user_dir/common_freq', selected_language + '.pkl'), 'rb') as f:
        common_tf = pickle.load(f)

    language_filter_tf = filter_top_percentile(language_tf, 95)
    filtered_counter = filter_by_ratio(language_filter_tf, common_tf, 0.95)

    output_dir = './wordlists-0.95'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{selected_language}.txt')

    # Save the filtered results to a text file
    with open(output_file, 'w') as f:
        for key, ratio in filtered_counter.items():
            f.write(f'{key}\n')


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

languages = {}
for tokenizer_id in tokenizer_to_languages.keys():
    tokenizer_path = os.path.join(root_path, tokenizer_id)

    for language in tokenizer_to_languages[tokenizer_id]:

        if os.path.exists(os.path.join(save_path, language + '.pkl')) and os.path.exists(os.path.join(corpus_path, language)):

            languages[language] = tokenizer_id

# Iterate over the languages and save the word lists (Super fast).
for selected_language, tokenizer_id in languages.items():
    print(selected_language, tokenizer_id)
    save(tokenizer_id, selected_language)






