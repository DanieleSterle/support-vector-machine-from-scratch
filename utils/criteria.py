import numpy as np
import pandas as pd

from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import PCA
import re


# Basic cleaning of tweets (remove links, mentions, hashtags, and whitespace)
def clean_text(text):
    if pd.isna(text):
        return ""
    
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)  # Remove mentions and hashtags
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    
    return text.lower()  # Convert to lowercase

# Handle negation by prefixing the next word with "NOT_"
def handle_negations(text):
    negations = [
        "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
        "isn't", "wasn't", "shouldn't", "wouldn't", "couldn't", "won't",
        "can't", "don't", "doesn't", "didn't"
    ]
    
    words = text.split()
    for i, word in enumerate(words):
        if word in negations and i + 1 < len(words):
            words[i + 1] = "NOT_" + words[i + 1]
    
    return " ".join(words)

# Tokenize and preprocess text data; optionally build or use a given vocabulary
def tokenization(df, text, min_freq=1, vocab=None):
    def lemmatize_word(word):
        lemmatizer = WordNetLemmatizer()
        if word.startswith('NOT_'):
            return 'NOT_' + lemmatizer.lemmatize(word[4:])
        return lemmatizer.lemmatize(word)

    def process_text_fully(text):
        if pd.isna(text) or text == "":
            return ""
        words = text.split()
        processed_words = []

        for word in words:
            if re.match(r'^[a-zA-Z_]+$', word):
                lemmatized = lemmatize_word(word)
                if lemmatized not in stop_words:
                    processed_words.append(lemmatized)
        return " ".join(processed_words)

    # Define stop words (words to ignore during processing)
    stop_words = {
        # Common English stop words
        "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", 
        "was", "were", "will", "with", "would", "have", "had", "this", "these", 
        "they", "them", "their", "there", "then", "than", "or", "do", "does", 
        "did", "doing", "done", "can", "could", "should", "shall", "may", 
        "might", "must", "ought", "i", "you", "your", "yours", "we", "our", 
        "ours", "she", "her", "hers", "him", "his", "us", "me", "my", "mine", 
        "am", "up", "down", "out", "off", "over", "under", "again", "further", 
        "once", "here", "where", "when", "why", "how", "all", "any", "both", 
        "each", "few", "more", "most", "other", "some", "such", "only", "own", 
        "same", "so", "through", "until", "while", "about", "against", "between", 
        "into", "during", "before", "after", "above", "below", "if", "because", 
        "since", "what", "which", "who", "whom", "whose", "get", "go", "going", 
        "gone", "got", "getting", "come", "coming", "came", "take", "taking", 
        "took", "taken", "put", "putting", "make", "making", "made", "see", 
        "seeing", "saw", "seen"
    }

    df_copy = df.copy()
    df_copy['processed_text'] = df_copy[text].apply(lambda x: handle_negations(x) if pd.notna(x) and x != "" else "")
    df_copy['fully_processed_text'] = df_copy['processed_text'].apply(process_text_fully)

    if vocab is None:
        # Build vocabulary
        words_series = df_copy['fully_processed_text'].str.split().explode().dropna()
        words_series = words_series[words_series != ""]
        word_counts = words_series.value_counts()

        tweet_counts = {}
        for word in word_counts.index:
            if word_counts[word] >= min_freq:
                tweet_count = df_copy['fully_processed_text'].apply(lambda x: word in x.split()).sum()
                tweet_counts[word] = tweet_count

        vocab = {
            word: [int(word_counts[word]), int(tweet_counts[word])]
            for word in word_counts.index if word_counts[word] >= min_freq
        }

    return df_copy, vocab

# Validate if a word contains only letters
def is_valid_word(word):
    if pd.isna(word):
        return False
    return bool(re.match(r'^[a-zA-Z]+$', str(word)))

# Compute TF-IDF score for a given word
def tf_idf(vocab, word, count, number_of_tweets):
    tf = 1 + np.log(count)
    idf = np.log(number_of_tweets / (vocab[word][1] + 1)) + 1
    return float(tf * idf)

# Generate TF-IDF matrix from dataset and vocabulary
def tf_idf_matrix(df, text, vocab):
    if vocab is None:
        clean_df, vocab = tokenization(df, text)
    else:
        clean_df, _ = tokenization(df, text, vocab=vocab)
        
    scof_x_i = []
    for _, tweet in clean_df.iterrows():
        tf_idf_vector = []
        text = tweet['fully_processed_text']

        for word in vocab.keys():
            words = text.split() if pd.notna(text) and text.strip() else []
            if word in words:
                count = words.count(word)
                tf_idf_vector.append(tf_idf(vocab, word, count, len(clean_df)))
            else:
                tf_idf_vector.append(0)
        scof_x_i.append(tf_idf_vector)

    return scof_x_i, vocab

def linear_kernel(X):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = np.dot(X[i], X[j])
    
    return K

def polynomial_kernel(X, coef0, degree):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = (np.dot(X[i], X[j]) + coef0) ** degree
    
    return K

def rbf_kernel(X, gamma):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            diff = X[i] - X[j]
            K[i, j] = np.exp(-gamma * np.dot(diff, diff))
    
    return K