import pandas as pd
import spacy
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Load the English NLP model from spaCy
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_trf")
print("Model loaded successfully.")

# Functions section
# Tokenize sentences and comments
def get_token_sent_count(text):
    doc = nlp(text)
    return len(doc), len(list(doc.sents))

# Common words per class
def get_most_common_words(class_name, top_n=20):
    texts = train_df[train_df[class_name] == 1]['comment_text'].tolist()
    word_list = []
    for doc in nlp.pipe(texts, batch_size=32):
        # Tokenize and filter out stop words and punctuation
        word_list.extend([
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and not token.is_stop
        ])
    return Counter(word_list).most_common(top_n)

# Main section
if __name__ == "__main__":
    print("Starting Toxic Comments Analysis...")
    # Define the path to the dataset
    DATA_PATH = "./resources/"

    # Load the datasets
    print("Loading datasets...")
    # train_df = pd.read_csv(DATA_PATH + "train.csv").head(5000)  # DEV
    train_df = pd.read_csv(DATA_PATH + "train.csv")  # PROD
    print(f"{len(train_df)} training samples loaded.")
    # test_df = pd.read_csv(DATA_PATH + "test.csv")
    # test_labels_df = pd.read_csv(DATA_PATH + "test_labels.csv")
    # sample_submission_df = pd.read_csv(DATA_PATH + "sample_submission.csv")

    # Columns to analyze
    target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # Process the training data
    print("Processing tokens and sentences...")
    texts = train_df["comment_text"].tolist()
    n_tokens = []
    n_sentences = []

    for doc in tqdm(nlp.pipe(texts, batch_size=32), total=len(texts)):
        n_tokens.append(len(doc))
        n_sentences.append(len(list(doc.sents)))

    train_df["n_tokens"] = n_tokens
    train_df["n_sentences"] = n_sentences

    print("Tokenization and sentence count completed.")
    print(train_df[["comment_text", "n_tokens", "n_sentences"]].head())

    # analyze class distribution
    print("Class distribution analysis...")
    class_distribution = train_df[target_columns].sum().sort_values(ascending=False)
    print("Number of comments per class:")
    print(class_distribution)

    # Visualisation of class distribution
    print("Visualizing class distribution...")
    plt.figure(figsize=(10, 5))
    sns.barplot(x=class_distribution.index, y=class_distribution.values)
    plt.title("Class Distribution of Toxic Comments (Number of Comments per Class)")
    plt.ylabel("Amount")
    plt.xlabel("Toxicity Class")
    plt.tight_layout()
    plt.show()

    # Get top words for each class
    print("Extracting most common words for each class...")
    top_words = get_most_common_words(train_df[train_df['toxic'] == 1]['comment_text'])
    print("Top words in toxic comments:")
    for word, count in top_words:
        print(f"{word:<15}: {count}")
