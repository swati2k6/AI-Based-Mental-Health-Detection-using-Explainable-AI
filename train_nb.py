import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

def train_nb(X_train, X_test, y_train, y_test):
    vectorizer = TfidfVectorizer(max_features=5000)
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    print("\nNaive Bayes Results:")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "models/nb_model.pkl")
    joblib.dump(vectorizer, "models/tfidf.pkl")

    return model, vectorizer, X_train_tfidf, X_test_tfidf, y_pred