import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Optimized Regex Patterns
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
PUNCT_PATTERN = re.compile('[%s]' % re.escape(string.punctuation))
NUM_PATTERN = re.compile(r'\d+')
WHITESPACE_PATTERN = re.compile(r'\s+')

class AetherDataProcessor:
    """
    Advanced text normalization engine for the Aether Sentinel.
    Standardizes raw signal input for neural ingestion.
    """
    def __init__(self):
        # Asset Synchronization
        resources = ['stopwords', 'wordnet', 'omw-1.4', 'punkt', 'punkt_tab']
        for res in resources:
            try:
                nltk.data.find(f'corpora/{res}' if res != 'punkt' else f'tokenizers/{res}')
            except LookupError:
                nltk.download(res, quiet=True)
            
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        """
        Runs a deep-cleaning sequence on raw text.
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Phase 1: Normalization
        text = text.lower()
        
        # Phase 2: Pattern Extraction
        text = URL_PATTERN.sub('', text)
        text = PUNCT_PATTERN.sub('', text)
        text = NUM_PATTERN.sub('', text)
        text = WHITESPACE_PATTERN.sub(' ', text).strip()
        
        # Phase 3: Morphological Analysis
        tokens = text.split() 
        cleaned_tokens = [
            self.lemmatizer.lemmatize(word) 
            for word in tokens if word not in self.stop_words and len(word) > 2
        ]
        
        return " ".join(cleaned_tokens)
