import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.preprocessing import clean_text
from src.train_nb import train_nb
from src.train_lstm import train_lstm
from src.train_bert import train_bert
from src.utils import plot_confusion
from src.explain import explain_model


# =========================
# 📊 METRICS FUNCTION
# =========================
def print_metrics(y_test, y_pred, model_name):
    print(f"\n===== {model_name} Results =====")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1-score :", f1_score(y_test, y_pred))


# =========================
# 📂 LOAD DATA
# =========================
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Rename columns
train_df = train_df.rename(columns={'sentence': 'text', 'sentiment': 'label'})
test_df = test_df.rename(columns={'sentence': 'text', 'sentiment': 'label'})

# Convert labels (positive → 0, negative → 1)
train_df['label'] = train_df['label'].apply(lambda x: 0 if x == 1 else 1)
test_df['label'] = test_df['label'].apply(lambda x: 0 if x == 1 else 1)

# Remove missing values
train_df = train_df.dropna(subset=['text'])
test_df = test_df.dropna(subset=['text'])

# =========================
# 🧹 PREPROCESSING
# =========================
train_df['clean_text'] = train_df['text'].apply(clean_text)
test_df['clean_text'] = test_df['text'].apply(clean_text)

X_train = train_df['clean_text']
y_train = train_df['label']

X_test = test_df['clean_text']
y_test = test_df['label']


# =========================
# 🤖 NAIVE BAYES
# =========================
nb_model, vectorizer, X_train_tfidf, X_test_tfidf, nb_pred = train_nb(
    X_train, X_test, y_train, y_test
)

print_metrics(y_test, nb_pred, "Naive Bayes")

# Confusion Matrix
plot_confusion(y_test, nb_pred)


# =========================
# 🧠 LSTM MODEL
# =========================
lstm_model, lstm_pred = train_lstm(X_train, X_test, y_train, y_test)

print_metrics(y_test, lstm_pred, "LSTM")


# =========================
# 🚀 BERT MODEL
# =========================
bert_model, bert_pred = train_bert(X_train, X_test, y_train, y_test)

print_metrics(y_test, bert_pred, "BERT")


# =========================
# 🔍 EXPLAINABILITY (SHAP)
# =========================
explain_model(nb_model, vectorizer, X_train_tfidf, X_test_tfidf)