import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from src.preprocessing import AetherDataProcessor
import os
import time

class AetherNeuralEngine:
    """
    Core Machine Learning Engine for the Aether Sentinel.
    Handles semantic vectorization and high-performance classification.
    """
    def __init__(self, model_path='models/model.pkl', vec_path='models/vectorizer.pkl'):
        self.model_path = model_path
        self.vec_path = vec_path
        # Optimized TF-IDF for high-dimensional semantic extraction
        self.vectorizer = TfidfVectorizer(
            max_features=5000, 
            ngram_range=(1, 1), 
            stop_words='english'
        )
        self.model = LogisticRegression(max_iter=2000, C=0.1, n_jobs=-1, solver='saga', class_weight='balanced')
        self.preprocessor = AetherDataProcessor()

    def prepare_data(self, true_csv, fake_csv):
        """
        Synchronizes raw data into a unified neural archive.
        """
        print("[DATA_SYNC] Accessing primary archives...")
        df_true = pd.read_csv(true_csv)
        df_fake = pd.read_csv(fake_csv)
        df_true['target'] = 1  
        df_fake['target'] = 0  
        
        df = pd.concat([df_true, df_fake]).reset_index(drop=True)
        df['total_text'] = df['title'].fillna('') + " " + df['text'].fillna('')
        
        print(f"[DATA_SYNC] Cleaning {len(df)} records for neural ingestion...")
        start_time = time.time()
        df['total_text'] = df['total_text'].apply(self.preprocessor.clean_text)
        print(f"[DATA_SYNC] Normalization complete in {time.time() - start_time:.2f}s")
        return df

    def train(self, df):
        """
        Calibrates neural weights based on the provided dataset.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            df['total_text'], df['target'], test_size=0.15, random_state=44
        )
        
        print("[TRAIN_INIT] Identifying semantic vectors...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        print("[TRAIN_INIT] Calibrating weight matrices...")
        start_time = time.time()
        self.model.fit(X_train_tfidf, y_train)
        print(f"[TRAIN_INIT] Calibration finished in {time.time() - start_time:.2f}s")
        
        # Performance Verification
        y_pred = self.model.predict(X_test_tfidf)
        y_prob = self.model.predict_proba(X_test_tfidf)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "report": classification_report(y_test, y_pred),
            "cm": confusion_matrix(y_test, y_pred)
        }
        
        # Archiving Core
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer, self.vec_path)
        print(f"[SYSTEM] Neural model archived to {self.model_path}")
        
        return metrics

    def predict(self, raw_text):
        """
        Runs a single text trace through the calibrated engine.
        """
        cleaned = self.preprocessor.clean_text(raw_text)
        vec = self.vectorizer.transform([cleaned])
        prob = self.model.predict_proba(vec)[0]
        return prob
