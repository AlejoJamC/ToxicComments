import pandas as pd
import spacy
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, roc_auc_score


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
    train_df = pd.read_csv(DATA_PATH + "train.csv").head(1000)  # DEV
    # train_df = pd.read_csv(DATA_PATH + "train.csv")  # PROD
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
    # plt.show()
    # Save the plot
    plt.savefig("./resources/class_distribution.png")

    # Get top words for each class
    print("Extracting most common words for each class...")
    top_words = get_most_common_words("toxic")
    print("Top words in toxic comments:")
    for word, count in top_words:
        print(f"{word:<15}: {count}")

    # TF-IDF Vectorization
    print("Starting TF-IDF vectorization...")
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(train_df["comment_text"])

    print("TF-IDF matrix shape:", X_tfidf.shape)
    print("First 10 features:", tfidf.get_feature_names_out()[:10])

    # Use only the 'toxic' column as the target for binary classification
    y = train_df["toxic"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    print("Logistic Regression Classification Report:")
    print(classification_report(y_test, y_pred_lr))
    print("AUC:", roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1]))

    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf))
    print("AUC:", roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]))

    print("\nTraining SVM (LinearSVC)...")
    svm_model = LinearSVC()
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    print("SVM Classification Report:")
    print(classification_report(y_test, y_pred_svm))


