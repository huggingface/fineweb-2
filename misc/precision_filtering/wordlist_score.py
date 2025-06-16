import os

def load_words_from_txt(folder_path):
    """
    Load words from text files in the given folder into a dictionary.
    
    folder_path: Path to the folder containing the language text files.
    
    Returns:
        A dictionary where keys are language names and values are lists of words.
    """
    language_dict = {}

    # List all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a .txt file
        if filename.endswith('.txt'):
            # Extract the language name from the filename (excluding the .txt extension)
            lang = filename[:-4]
            word_list = []

            # Open the file and read the words
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                for line in file:
                    word = line.strip()  # Remove any leading/trailing whitespace
                    if word:  # Skip empty lines
                        word_list.append(word)

            # Store the list of words for the current language in the dictionary
            language_dict[lang] = set(word_list)
    
    return language_dict

# Example usage
folder_path = 'wordlists-0.95'  # Replace with your folder path
loaded_language_dict = load_words_from_txt(folder_path)


from datatrove.utils.text import split_into_words, simplify_text, TextNormConfig

norm_config = TextNormConfig(
    lowercase=False,
    norm_numbers=False,
    norm_weekdays=False,
    norm_monthnames=False,
    remove_punctuation=True,
    norm_unicode_diacritics=False,
    norm_whitespace=True,
)

def filter_score(text, lang):
    text = simplify_text(str(text).strip(), norm_config)
    # Split the text into words and convert it to a set
    words = set(split_into_words(text, lang))
    # Get the list of words for the specified language and convert it to a set
    lang_words = loaded_language_dict[lang]  # Convert to set for faster lookups

    # Count how many words from the text appear in the language's word list
    matching_words = len(words & lang_words)  # Use set intersection to count matches

    # Calculate the normalized score (between 0 and 1)
    score = matching_words / len(words) if words else 0  # Avoid division by zero if text is empty
    return score

# Example usage
text = "Servus, i bin da Sepp aus Minga. I leb scho a Zeitlang do und gfoid ma recht guad. I sauf gern a Hoibe und geh am Wochenende auf’d Wiesn, wenn’s geht. 🍺"

score = filter_score(text, 'bar_Latn')
threshold = 0  # Adjust as needed based on your use case

if score > threshold:
    print("Label is correct.")
else:
    print("Label is incorrect (possible contamination).")
