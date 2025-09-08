import pandas as pd
import numpy as np
import re
import pickle
import math
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Data Preprocessing Functions
def remove_html(sentence):
    pattern = re.compile('<.*?>')
    return re.sub(pattern, ' ', sentence)

def remove_punctuations(sentence):
    return re.sub('[^a-zA-Z]', ' ', sentence)

def strip_lower(sentence):
    cleaned_text = sentence.lower().strip()
    cleaned_text = re.sub(' +', ' ', cleaned_text)
    return cleaned_text

def take_only_distinct_words(sentence):
    word_tokens = sentence.split()
    seen = set()
    cleaned_text = []
    for word in word_tokens:
        if word not in seen:
            cleaned_text.append(word)
            seen.add(word)
    return ' '.join(cleaned_text)

def remove_stopwords(sentence):
    stop_words = set(stopwords.words('english'))
    negative_words = ["not", "no", "nor", "never", "don't", "didn't", "doesn't", "isn't", "wasn't", "shouldn't", "wouldn't", "couldn't", "won't", "haven't", "hasn't", "hadn't", "mightn't", "mustn't", "shan't"]
    stop_words = stop_words - set(negative_words)
    word_tokens = sentence.split()
    cleaned_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(cleaned_text)

def stemming(sentence):
    porter_stemmer = PorterStemmer()
    word_tokens = sentence.split()
    cleaned_text = [porter_stemmer.stem(word) for word in word_tokens]
    return ' '.join(cleaned_text)

def data_preprocessing(sentence):
    sentence = remove_html(sentence)
    sentence = remove_punctuations(sentence)
    sentence = strip_lower(sentence)
    sentence = take_only_distinct_words(sentence)
    sentence = remove_stopwords(sentence)
    sentence = stemming(sentence)
    return sentence

# Text to Numerical Data Conversion
def text_to_bow(corpus, save_path='count_vectorizer.pkl'):
    cv_object = CountVectorizer()
    X_bow = cv_object.fit_transform(corpus)
    with open(save_path, 'wb') as f:
        pickle.dump(cv_object, f)
    return X_bow

def text_to_bow_test(corpus, load_path='count_vectorizer.pkl'):
    with open(load_path, 'rb') as f:
        loaded_cv_object = pickle.load(f)
    return loaded_cv_object.transform(corpus)

def text_to_tfidf(corpus, save_path='tfidf_vectorizer.pkl'):
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(corpus)
    with open(save_path, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    return X_tfidf

def text_to_tfidf_test(corpus, load_path='tfidf_vectorizer.pkl'):
    with open(load_path, 'rb') as f:
        loaded_tfidf_vectorizer = pickle.load(f)
    return loaded_tfidf_vectorizer.transform(corpus)

# Model Building Function(got best accuracy with these hyperparameters)
#random forest classifier is giving the best results so we will save this model for future use
# saving the random forest model in pickle file for future use
def train_model(X, y, save_path='rf_model.pkl'):
    model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=1, max_features='sqrt', bootstrap=True, class_weight=None, criterion='gini')
    model.fit(X, y)
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    return model

# Model Prediction Function
def predict_sentiment(text, model_path='rf_model.pkl', vectorizer_path='count_vectorizer.pkl'):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    # Preprocess the input text
    cleaned_text = data_preprocessing(text)
    # Transform and predict
    X = vectorizer.transform([cleaned_text])  # Wrap in list
    pred = model.predict(X)[0]
    return "Positive" if pred == 1 else "Negative"
# Main Pipeline Example
if __name__ == "__main__":
    # Load data
    df = pd.read_csv("Reviews.csv")
    df = df.sample(frac=0.1, random_state=1)
    df['SentimentPolarity'] = df['Score'].apply(lambda x: 'Positive' if x > 3 else 'Negative')
    df['Class_Labels'] = df['SentimentPolarity'].apply(lambda x: 1 if x == 'Positive' else 0)
    df = df.drop_duplicates(subset=["UserId", "ProfileName", "Time", "Text"], keep='first')
    df = df[df.HelpfulnessNumerator <= df.HelpfulnessDenominator]
    sampled_dataset = df.drop(labels=['Id','ProductId', 'UserId', 'Score', 'ProfileName','HelpfulnessNumerator', 'HelpfulnessDenominator','Summary'], axis=1)
    sampled_dataset = sampled_dataset.sort_values('Time', axis=0, ascending=False)
    sampled_dataset = sampled_dataset.reset_index(drop=True)
    sampled_dataset['Cleaned_Text'] = sampled_dataset['Text'].apply(data_preprocessing)
    sampled_dataset = sampled_dataset[['Time','Cleaned_Text','Class_Labels']]
    X = sampled_dataset['Cleaned_Text']
    y = sampled_dataset['Class_Labels']
    split = math.floor(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # Convert text to numerical data
    X_bow = text_to_bow(X_train)
    X_test_bow = text_to_bow_test(X_test)

    # Train and save model
    model = train_model(X_bow, y_train)

    # Predict on test data
    y_pred = predict_sentiment(X_test, model_path='rf_model.pkl', vectorizer_path='count_vectorizer.pkl')

    # Print accuracy
    from sklearn.metrics import classification_report
    print("Classification Report on test data:\n", classification_report(y_test, y_pred))