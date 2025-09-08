import pandas as pd
import re
import math
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

# Data Preprocessing
def preprocess_text(text):
    text = re.sub('<.*?>', ' ', text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().strip()
    text = re.sub(' +', ' ', text)
    stop_words = set(stopwords.words('english'))
    negative_words = {"not", "no", "nor", "never", "don't", "didn't", "doesn't", "isn't", "wasn't", "shouldn't", "wouldn't", "couldn't", "won't", "haven't", "hasn't", "hadn't", "mightn't", "mustn't", "shan't"}
    stop_words = stop_words - negative_words
    words = [w for w in text.split() if w not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]
    seen = set()
    distinct_words = []
    for word in words:
        if word not in seen:
            distinct_words.append(word)
            seen.add(word)
    return ' '.join(distinct_words)

# Load and preprocess data
df = pd.read_csv("Reviews.csv")
df = df.sample(frac=0.1, random_state=1)
df['SentimentPolarity'] = df['Score'].apply(lambda x: 'Positive' if x > 3 else 'Negative')
df['Class_Labels'] = df['SentimentPolarity'].apply(lambda x: 1 if x == 'Positive' else 0)
df = df.drop_duplicates(subset=["UserId", "ProfileName", "Time", "Text"], keep='first')
df = df[df.HelpfulnessNumerator <= df.HelpfulnessDenominator]
sampled_dataset = df.drop(labels=['Id','ProductId', 'UserId', 'Score', 'ProfileName','HelpfulnessNumerator', 'HelpfulnessDenominator','Summary'], axis=1)
sampled_dataset = sampled_dataset.sort_values('Time', axis=0, ascending=False)
sampled_dataset = sampled_dataset.reset_index(drop=True)
sampled_dataset['Cleaned_Text'] = sampled_dataset['Text'].apply(preprocess_text)
sampled_dataset = sampled_dataset[['Time','Cleaned_Text','Class_Labels']]
X = sampled_dataset['Cleaned_Text']
y = sampled_dataset['Class_Labels']
split = math.floor(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Convert text to numerical data
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Try classifiers
results = {}

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_bow, y_train)
y_pred_nb = nb_model.predict(X_test_bow)
results['Naive Bayes'] = {
    'accuracy': accuracy_score(y_test, y_pred_nb),
    'report': classification_report(y_test, y_pred_nb)
}

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_bow, y_train)
y_pred_dt = dt_model.predict(X_test_bow)
results['Decision Tree'] = {
    'accuracy': accuracy_score(y_test, y_pred_dt),
    'report': classification_report(y_test, y_pred_dt)
}

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_bow, y_train)
y_pred_rf = rf_model.predict(X_test_bow)
results['Random Forest'] = {
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'report': classification_report(y_test, y_pred_rf)
}

# XGBoost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_bow, y_train)
y_pred_xgb = xgb_model.predict(X_test_bow)
results['XGBoost'] = {
    'accuracy': accuracy_score(y_test, y_pred_xgb),
    'report': classification_report(y_test, y_pred_xgb)
}

# Print results
for name, res in results.items():
    print(f"\n{name} Accuracy: {res['accuracy']:.4f}")
    print(f"{name} Classification Report:\n{res['report']}")

# Select best model
best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\nBest Model: {best_model[0]} with Accuracy: {best_model[1]['accuracy']:.4f}")