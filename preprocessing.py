import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# 🔹 Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)        # remove URLs
    text = re.sub(r'@\w+', '', text)           # remove mentions
    text = re.sub(r'#\w+', '', text)           # remove hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)    # remove special chars
    
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    
    return " ".join(words)

# 🔹 Add sentiment feature (important for your paper)
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = sia.polarity_scores(text)
    return score['compound']

# 🔹 Full preprocessing pipeline
def preprocess_dataset(file_path):
    df = pd.read_csv(file_path)

    # Clean text
    df['clean_text'] = df['text'].apply(clean_text)

    # Add sentiment score
    df['sentiment'] = df['clean_text'].apply(get_sentiment)

    # Remove empty rows
    df = df[df['clean_text'].str.strip() != ""]

    print("Dataset shape:", df.shape)
    print(df.head())

    return df