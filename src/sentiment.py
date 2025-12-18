import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
import logging
import os
from src.utils import get_device

logger = logging.getLogger(__name__)

class SentimentAnalyzerModule:
    def __init__(self, model_name: str = "xlm-roberta-base", lora_path: str = "models/xlmroberta_checkpoints"): #sentiment_lora
        self.device = get_device()
        
        # Definisci il percorso per la cache locale dei modelli e path assoluto LoRA
        self.models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
        self.lora_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', lora_path))
        
        logger.info(f"Inizializzazione Sentiment Module su {self.device}...")
        
        try:
            # 1. Carica Tokenizer e Configurazione Base
            logger.info(f"Caricamento modello base: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.models_dir)
            
            # Label map standard per xlm-roberta-base trained con il nostro script
            id2label = {0: "negative", 1: "neutral", 2: "positive"}
            label2id = {"negative": 0, "neutral": 1, "positive": 2}

            # 2. Carica Modello Base
            self.base_model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=3,
                id2label=id2label,
                label2id=label2id,
                cache_dir=self.models_dir
            )
            
            # 3. Carica LoRA se esiste
            if os.path.exists(self.lora_path) and os.path.exists(os.path.join(self.lora_path, "adapter_config.json")):
                logger.info(f"Trovati pesi LoRA in {self.lora_path}. Caricamento...")
                self.model = PeftModel.from_pretrained(self.base_model, self.lora_path)
                self.model.to(self.device)
                logger.info("Modello LoRA caricato con successo.")
            else:
                logger.warning(f"Pesi LoRA non trovati in {self.lora_path}. Uso il modello base (non finetunato).")
                self.model = self.base_model.to(self.device)

            # Pipeline non supporta nativamente PeftModel in modo pulito a volte, usiamo inferenza manuale o wrap
            # Ma per semplicità usiamo la chiamata diretta al modello nel metodo analyze
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Errore caricamento modello: {e}")
            raise e

    def analyze(self, text: str):
        """
        Analizza una singola stringa.
        Ritorna: {'label': 'positive'/'negative'/'neutral', 'score': float}
        """
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Ottieni la classe con probabilità maggiore
            score, class_id = torch.max(probabilities, dim=-1)
            label = self.model.config.id2label[class_id.item()]
            
            return {'label': label, 'score': score.item()}
            
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