import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

class AetherForensicExplainer:
    """
    Advanced Forensic Explainer for the Aether Sentinel.
    Provides SHAP-based feature attribution and linguistic pattern analysis.
    """
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer
        # High-precision prediction function for semantic attribution
        self.predict_fn = lambda x: self.model.predict_proba(self.vectorizer.transform(x))[:, 1]
        self.explainer = shap.Explainer(self.predict_fn, masker=shap.maskers.Text(tokenizer=r"\W+"))

    def get_local_explanation(self, text_list):
        """
        Explains a specific instance of news using SHAP.
        """
        try:
            shap_values = self.explainer(text_list)
            return shap_values
        except Exception as e:
            print(f"[ERROR] SHAP Trace Failure: {e}")
            return None

    def get_linguistic_audit(self, text):
        """
        Aether Linguistic Intelligence Agent.
        Scans text for syntactic anomalies and emotional triggers.
        """
        if not text:
            return {"status": "error", "report": ["No input provided."]}
            
        words = text.split()
        word_count = len(words)
        caps_words = [w for w in words if w.isupper() and len(w) > 1]
        exclamations = text.count('!')
        question_marks = text.count('?')
        
        # Forensic Scoring logic
        sensationalism_score = (len(caps_words) / (word_count + 1)) * 100
        urgency_score = (exclamations / (word_count + 1)) * 100
        speculation_ratio = (question_marks / (word_count + 1)) * 100
        
        report = []
        if sensationalism_score > 15:
            report.append(f"⚠️ HIGH SENSATIONALISM: {len(caps_words)} words are in ALL CAPS. Pattern matching clickbait signatures.")
        if urgency_score > 5:
            report.append(f"🚨 URGENCY MARKERS: High frequency of exclamation marks ({exclamations}). Likely emotional provocation.")
        if speculation_ratio > 4:
            report.append("❓ SPECULATIVE TONE: High density of interrogation marks. Suggests unverified inquiry patterns.")
        if word_count < 35:
            report.append("📉 LOW SUBSTANCE: Article length below forensic threshold. Legitimate reports typically provide greater context.")
            
        final_assessment = "Standard"
        if len(report) >= 2:
            final_assessment = "Highly Suspicious"
        elif len(report) == 1:
            final_assessment = "Caution Advised"
            
        return {
            "assessment": final_assessment,
            "report": report if report else ["Linguistic patterns verified within professional benchmarks."],
            "stats": {
                "Sensationalism": f"{sensationalism_score:.1f}%",
                "Urgency": f"{urgency_score:.1f}%",
                "Speculation": f"{speculation_ratio:.1f}%"
            }
        }

    def get_related_intel(self, query_text, dataset_df):
        """
        Synchronizes with historic archives to find semantic overlaps.
        """
        if not query_text or dataset_df is None:
            return []
            
        from sklearn.metrics.pairwise import cosine_similarity
        
        query_vec = self.vectorizer.transform([str(query_text)])
        sample_size = min(len(dataset_df), 4000) # Increased precision
        subset = dataset_df.sample(sample_size, random_state=42)
        
        dataset_vecs = self.vectorizer.transform(subset['total_text'].astype(str))
        similarities = cosine_similarity(query_vec, dataset_vecs).flatten()
        
        # Extract top forensic matches
        top_indices = similarities.argsort()[-5:][::-1]
        
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score < 0.08: continue 
            
            row = subset.iloc[idx]
            results.append({
                "title": str(row['title']).strip(),
                "content": str(row['text']).strip()[:350] + "...",
                "similarity": f"{score:.1%}",
                "status": "Verified" if row['target'] == 1 else "Flagged",
                "label": "REAL" if row['target'] == 1 else "FAKE"
            })
        return results

    def get_global_importance(self):
        """
        Extracts global neural weightings for authentic vs anomalous markers.
        """
        coefs = self.model.coef_[0]
        features = self.vectorizer.get_feature_names_out()
        
        importance_df = pd.DataFrame({'word': features, 'coefficient': coefs})
        
        real_indicators = importance_df.sort_values(by='coefficient', ascending=False).head(25).copy()
        fake_indicators = importance_df.sort_values(by='coefficient', ascending=True).head(25).copy()
        
        real_indicators.columns = ['AUTHENTIC_MARKER', 'WEIGHT']
        fake_indicators.columns = ['ANOMALY_TRIGGER', 'WEIGHT']
        
        return real_indicators, fake_indicators
