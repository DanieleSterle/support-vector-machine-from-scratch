import numpy as np
import pandas as pd

from nltk.stem import WordNetLemmatizer
import re

# Basic cleaning of phrases: remove URLs, mentions, hashtags, extra spaces, lowercase
def clean_text(text):
    if pd.isna(text):
        return ""
    
    # Remove URLs from text
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove Twitter mentions (@username) and hashtags (#tag)
    text = re.sub(r'@\w+|#\w+', '', text)
    # Replace multiple spaces with a single space and strip leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Convert to lowercase for normalization
    return text.lower()


# Handle negations by prefixing the following word with "NOT_"
def handle_negations(text):
    negations = [
        "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
        "isn't", "wasn't", "shouldn't", "wouldn't", "couldn't", "won't",
        "can't", "don't", "doesn't", "didn't"
    ]
    
    words = text.split()
    for i, word in enumerate(words):
        # If a negation word is found, prefix the next word with "NOT_"
        if word in negations and i + 1 < len(words):
            words[i + 1] = "NOT_" + words[i + 1]
    
    return " ".join(words)


# Tokenize, lemmatize, remove stopwords; optionally build or use a vocabulary
def tokenization(df, text, min_freq=1, vocab=None):

    def lemmatize_word(word):
        lemmatizer = WordNetLemmatizer()
        # Preserve negation prefix during lemmatization
        if word.startswith('NOT_'):
            return 'NOT_' + lemmatizer.lemmatize(word[4:])
        return lemmatizer.lemmatize(word)

    def process_text_fully(text):
        if pd.isna(text) or text == "":
            return ""
        words = text.split()
        processed_words = []

        for word in words:
            # Only keep alphabetic words or words with underscores (like NOT_word)
            if re.match(r'^[a-zA-Z_]+$', word):
                lemmatized = lemmatize_word(word)
                # Filter out stop words
                if lemmatized not in stop_words:
                    processed_words.append(lemmatized)
        # Recombine processed words into a cleaned string
        return " ".join(processed_words)

    # Stop words to ignore during tokenization
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
    # Apply negation handling to each phrase
    df_copy['processed_text'] = df_copy[text].apply(
        lambda x: handle_negations(x) if pd.notna(x) and x != "" else ""
    )
    # Apply lemmatization and stop-word removal
    df_copy['fully_processed_text'] = df_copy['processed_text'].apply(process_text_fully)

    if vocab is None:
        # Build vocabulary based on minimum frequency across phrases
        words_series = df_copy['fully_processed_text'].str.split().explode().dropna()
        words_series = words_series[words_series != ""]
        word_counts = words_series.value_counts()

        phrase_counts = {}
        for word in word_counts.index:
            if word_counts[word] >= min_freq:
                # Count in how many phrases the word appears
                phrase_count = df_copy['fully_processed_text'].apply(
                    lambda x: word in x.split()
                ).sum()
                phrase_counts[word] = phrase_count

        # Vocabulary structure: {word: [total_count, phrase_count]}
        vocab = {
            word: [int(word_counts[word]), int(phrase_counts[word])]
            for word in word_counts.index if word_counts[word] >= min_freq
        }

    return df_copy, vocab


# Validate that a word contains only letters
def is_valid_word(word):
    if pd.isna(word):
        return False
    return bool(re.match(r'^[a-zA-Z]+$', str(word)))


# Compute TF-IDF for a single word
def tf_idf(vocab, word, count, number_of_phrases):
    # Term frequency with log scaling
    tf = 1 + np.log(count)
    # Inverse document frequency with smoothing
    idf = np.log(number_of_phrases / (vocab[word][1] + 1)) + 1
    return float(tf * idf)


# Generate TF-IDF matrix for dataset
def tf_idf_matrix(df, text, vocab):

    if vocab is None:
        clean_df, vocab = tokenization(df, text)
    else:
        # Use existing vocabulary for consistency
        clean_df, _ = tokenization(df, text, vocab=vocab)
        
    scof_x_i = []
    for _, phrase in clean_df.iterrows():
        tf_idf_vector = []
        text = phrase['fully_processed_text']

        # Compute TF-IDF for each word in vocabulary
        for word in vocab.keys():
            words = text.split() if pd.notna(text) and text.strip() else []
            if word in words:
                count = words.count(word)
                tf_idf_vector.append(tf_idf(vocab, word, count, len(clean_df)))
            else:
                tf_idf_vector.append(0)
        scof_x_i.append(tf_idf_vector)

    return scof_x_i, vocab


# Linear kernel K(x, y) = x^T y
def linear_kernel(X):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))

    # Compute pairwise dot products
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = np.dot(X[i], X[j])
    
    return K


# Polynomial kernel K(x, y) = (x^T y + coef0)^degree
def polynomial_kernel(X, coef0, degree):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            # Apply polynomial transformation to dot product
            K[i, j] = (np.dot(X[i], X[j]) + coef0) ** degree
    
    return K


# RBF kernel K(x, y) = exp(-gamma * ||x-y||^2)
def rbf_kernel(X, gamma):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            # Squared Euclidean distance
            diff = X[i] - X[j]
            K[i, j] = np.exp(-gamma * np.dot(diff, diff))
    
    return K