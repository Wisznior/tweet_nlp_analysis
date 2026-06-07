import re
import logging
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\b(realdonaldtrump|donaldtrump|trump2016)\b', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    stop_words = set(stopwords.words('english'))
    
    tokens = [t for  t in text.split() if t not in stop_words]
    return ' '.join(tokens)

def create_target(df: pd.DataFrame) -> pd.Series:
    q33 =df['retweets'].quantile(0.33)
    q66 =df['retweets'].quantile(0.66)

    logger.info(f"RT quantiles - class 0: 0-{q33:.0f}, class 1: {q33:.0f}-{q66:.0f}, class 2: {q66:.0f}+")

    def label(x):
        if x <= q33:
            return 0
        elif x <= q66:
            return 1
        else:
            return 2
    
    return df['retweets'].apply(label)

def build_vectorizer(method: str = "tfidf", ngram_range: tuple = (1, 2), max_features: int = 5000):
    params = dict(max_features=max_features, ngram_range=ngram_range, min_df=5, max_df=0.95)
    if method == "tfidf":
        return TfidfVectorizer(**params)
    if method == "bow":
        return CountVectorizer(**params)
    raise ValueError(f"Unknown method '{method}'. Use 'tfidf' or 'bow'.")


def build_tfidf(texts, max_features: int = 5000) -> tuple:
    vectorizer = build_vectorizer("tfidf", (1, 2), max_features)

    X = vectorizer.fit_transform(texts)

    logger.info(f"TF-IDF: {X.shape[0]} tweets, {X.shape[1]} features")

    return X, vectorizer

def load_and_clean(db_engine) -> tuple:
    logger.info("Loadingdata from database")
    df = pd.read_sql(
        "SELECT content, retweets, date FROM raw_tweets",
        db_engine
    )
    logger.info(f"Loaded {len(df)} tweets")

    df = df.dropna(subset=['content', 'retweets'])
    df['clean'] = df['content'].apply(clean_text)

    before = len(df)
    df = df[df['clean'].str.len() > 0]
    logger.info(f"Removed {before - len(df)} empty tweets after cleaning, {len(df)} tweets remaining")

    y = create_target(df)
    return df, y


def build_metadata_features(df) -> tuple:
    content = df['content'].astype(str)
    dates = pd.to_datetime(df['date'], errors='coerce')

    features = pd.DataFrame({
        'length': content.str.len(),
        'word_count': content.str.split().str.len(),
        'hour': dates.dt.hour,
        'dayofweek': dates.dt.dayofweek,
        'has_url': content.str.contains('http', case=False, regex=False).astype(int),
    })

    feature_names = ['length', 'word_count', 'hour', 'dayofweek', 'has_url']
    X_meta = features[feature_names].to_numpy(dtype=float)

    logger.info(f"Metadata features: {X_meta.shape[0]} rows, {X_meta.shape[1]} features")
    return X_meta, feature_names


def prepare_data(db_engine, vectorizer_path: str = 'data/tfidf_vectorizer.pkl'):
    df, y = load_and_clean(db_engine)

    logger.info("Building TF-IDF matrix")
    X, vectorizer = build_tfidf(df['clean'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state = 42, stratify = y
    )

    logger.info(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    logger.info(f"Class distribution (train): {y_train.value_counts().sort_index().to_dict()}")

    joblib.dump(vectorizer, vectorizer_path)
    logger.info(f"Vectorizer saved to: {vectorizer_path}")

    feature_names = vectorizer.get_feature_names_out()

    return X_train, X_test, y_train, y_test, feature_names