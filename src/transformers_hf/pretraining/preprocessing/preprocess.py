import re
import string
from langdetect import detect
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import json 

# Load the config file
config_path = "./config/pretrain_config.json"
with open(config_path, "r") as f:
    config = json.load(f)

datasetpath = config['dataset_source']
# Example: datasetpath = "./data/stackexchange/stackoverflow/stackoverflow.txt"

savepath = config["preprocessed_dataset_path"]

# Download NLTK resources
import nltk
nltk.download("punkt")
nltk.download("stopwords")


def clean_text(text):
    """
    Clean and preprocess text data.
    """
    # Remove extra whitespaces
    # eg: "hello    world" -> "hello world"
    text = re.sub(r"\s+", " ", text).strip()
    
    # Remove special characters (retain punctuation if needed)
    # eg: "hello, world!" -> "hello world"
    text = re.sub(r"[^\w\s" + re.escape(string.punctuation) + "]", "", text)
    
    #remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    
    # Convert to lowercase
    # eg: "Hello World" -> "hello world"
    text = text.lower()

    #remove 
    
    return text


def detect_and_filter_language(text, target_lang="en"):
    """
    Detect language and filter by target language.
    eg: "Bonjour le monde" -> None
    """
    try:
        detected_lang = detect(text)
        if detected_lang == target_lang:
            return text
        else:
            return None
    except Exception as e:
        return None


def tokenize_text(text):
    """
    Tokenize text into words.
    eg: "Hello World" -> ["Hello", "World"]
    """
    tokens = word_tokenize(text)
    return tokens


def remove_stopwords(tokens, language="english"):
    """
    Remove stopwords from tokenized text.
    eg: ["Hello", "World"] -> ["Hello"]
    """
    stop_words = set(stopwords.words(language))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return filtered_tokens


def preprocess_text(text, target_language="en", remove_stop_words=True):
    """
    Complete preprocessing pipeline.
    """
    # Language detection and filtering
    text = detect_and_filter_language(text, target_lang=target_language)
    if not text:
        return None  # Skip non-target language texts
    
    # Clean text
    text = clean_text(text)
    
    # Tokenize
    tokens = tokenize_text(text)
    
    # Remove stopwords if required
    if remove_stop_words:
        tokens = remove_stopwords(tokens)
    
    return " ".join(tokens)


def split_into_paragraphs(text):
    """
    Split text into paragraphs based on double line breaks.
    """
    paragraphs = re.split(r'\n\s*\n', text)
    # Remove empty paragraphs
    paragraphs = [para.strip() for para in paragraphs if para.strip()]
    return paragraphs


def save_preprocessed_text(preprocessed_lines, output_file):
    """
    Save processed text lines to a file, ensuring each is on a new line.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for line in preprocessed_lines:
            if line:  # Ensure line is not None or empty
                f.write(line + "\n")


# Example usage
if __name__ == "__main__":
    # Load raw text
    with open(datasetpath, "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    # Split raw text into paragraphs (or other chunks)
    paragraphs = split_into_paragraphs(raw_text)
    
    print(f"Total paragraphs found: {len(paragraphs)}")
    
    # Preprocess each paragraph
    preprocessed_texts = []
    for i, para in enumerate(paragraphs):
        processed = preprocess_text(para)
        if processed:
            preprocessed_texts.append(processed)
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} paragraphs")
    
    print(f"Total preprocessed paragraphs: {len(preprocessed_texts)}")
    
    # Save the preprocessed text, each paragraph on a new line
    savepath = config["preprocessed_dataset_path"]
    save_preprocessed_text(preprocessed_texts, savepath)
    
    print(f"Preprocessed data saved to {savepath}")
