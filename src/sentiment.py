import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import logging
import os
from src.utils import get_device

logger = logging.getLogger(__name__)

class SentimentAnalyzerModule:
    def __init__(self, model_name: str = "MilaNLProc/feel-it-italian-sentiment"):
        self.device = get_device()
        
        # Definisci il percorso per la cache locale dei modelli
        self.models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
        
        logger.info(f"Caricamento modello Sentiment su {self.device}...")
        logger.info(f"Cache modelli locale: {self.models_dir}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.models_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=self.models_dir).to(self.device)
            
            self.classifier = pipeline(
                "text-classification", 
                model=self.model, 
                tokenizer=self.tokenizer, 
                device=self.device,
                return_all_scores=True 
            )
            logger.info("Modello Sentiment caricato con successo.")
        except Exception as e:
            logger.error(f"Errore caricamento modello {model_name}: {e}")
            raise e

    def analyze(self, text: str):
        """
        Analizza una singola stringa.
        Ritorna: {'label': 'positive'/'negative', 'score': float}
        """
        try:
            results = self.classifier(text, truncation=True, max_length=512)
            scores = results[0]
            best_score = max(scores, key=lambda x: x['score'])
            return best_score
        except Exception as e:
            logger.error(f"Errore analisi sentiment: {e}")
            return {'label': 'error', 'score': 0.0}

    def analyze_batch(self, texts: list):
        """
        Analizza una lista di testi.
        """
        results = []
        for text in texts:
            results.append(self.analyze(text))
        return results