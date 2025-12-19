import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
import logging
import os
from src.utils import get_device


def _infer_num_labels_from_adapter(adapter_path: str) -> int | None:
    """
    Prova a dedurre il numero di classi dall'adapter LoRA
    leggendo la shape del classificatore salvato nel safetensor.
    Restituisce None se non è possibile inferire.
    """
    try:
        from safetensors.torch import load_file

        safetensor_path = os.path.join(adapter_path, "adapter_model.safetensors")
        if not os.path.exists(safetensor_path):
            return None

        state = load_file(safetensor_path, device="cpu")

        # Prova a cercare il proiettore finale
        for key, tensor in state.items():
            if key.endswith("out_proj.weight") and tensor.ndim == 2:
                return tensor.shape[0]
            if key.endswith("out_proj.bias") and tensor.ndim == 1:
                return tensor.shape[0]
    except Exception as exc:  # noqa: BLE001 - loggiamo e torniamo None
        logger.warning(f"Impossibile inferire num_labels dall'adapter: {exc}")

    return None

logger = logging.getLogger(__name__)

class SentimentAnalyzerModule:
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        lora_path: str = "models/sentiment_lora",
        fallback_lora_paths: tuple = ("models/xlmroberta_checkpoints",),
    ):
        """
        Inizializza l'analizzatore di sentiment usando un adapter LoRA se disponibile.
        - model_name: modello base HF
        - lora_path: percorso predefinito per l'adapter LoRA fine-tunato (quello del notebook)
        - fallback_lora_paths: percorsi aggiuntivi provati in caso il principale non esista
        """

        self.device = get_device()

        # Percorsi assoluti
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.models_dir = os.path.join(base_dir, 'models')

        # Consenti override via env var (es. SENTIMENT_LORA_PATH) e prova fallback sequenziale
        env_override = os.getenv("SENTIMENT_LORA_PATH")
        candidate_paths = [env_override] if env_override else []
        candidate_paths.append(lora_path)
        candidate_paths.extend(list(fallback_lora_paths))

        self.lora_path = None
        for path in candidate_paths:
            abs_path = os.path.abspath(os.path.join(base_dir, path))
            if os.path.exists(os.path.join(abs_path, "adapter_config.json")):
                self.lora_path = abs_path
                break

        logger.info(f"Inizializzazione Sentiment Module su {self.device}...")

        try:
            # 1) Leggi config dell'adapter (se presente) per recuperare info base
            base_model_source = model_name
            adapter_num_labels = None
            peft_label_map = None

            if self.lora_path:
                try:
                    peft_cfg = PeftConfig.from_pretrained(self.lora_path)
                    base_model_source = peft_cfg.base_model_name_or_path or base_model_source
                    adapter_num_labels = getattr(peft_cfg, "num_labels", None)
                    if getattr(peft_cfg, "id2label", None):
                        peft_label_map = peft_cfg.id2label
                except Exception as exc:  # noqa: BLE001 - informativo
                    logger.warning(f"Impossibile leggere peft config: {exc}")

            if adapter_num_labels is None and self.lora_path:
                adapter_num_labels = _infer_num_labels_from_adapter(self.lora_path)

            # 2) Carica tokenizer (se esiste un adapter, usa il tokenizer salvato lì)
            tokenizer_source = self.lora_path if self.lora_path else base_model_source
            logger.info(f"Caricamento tokenizer da: {tokenizer_source}")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, cache_dir=self.models_dir)

            # 3) Determina l'ordine delle label
            mental_health_labels = [
                "anxiety",
                "bipolar",
                "depression",
                "normal",
                "personality disorder",
                "stress",
                "suicidal",
            ]
            generic_sentiment = ["negative", "neutral", "positive"]
            mental_grouped = ["depression", "Light", "Normal", "Serious"]  # vedi mapping nel notebook EDA

            label_presets = {
                3: generic_sentiment,
                4: mental_grouped,  # vedi mapping nel notebook EDA
                7: mental_health_labels,
            }

            labels_order = mental_grouped
            
            id2label = {i: lbl for i, lbl in enumerate(labels_order)}
            label2id = {lbl: i for i, lbl in enumerate(labels_order)}
            logger.info(f"Classi impostate: {labels_order}")

            # 4) Modello base
            self.base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_source,
                num_labels=len(id2label),
                id2label=id2label,
                label2id=label2id,
                cache_dir=self.models_dir,
            )

            # 5) Applica LoRA se presente
            if self.lora_path:
                logger.info(f"Trovato adapter LoRA in {self.lora_path}. Caricamento...")
                self.model = PeftModel.from_pretrained(self.base_model, self.lora_path)
                self.model.to(self.device)
                logger.info("Modello LoRA caricato con successo.")
            else:
                logger.warning("Adapter LoRA non trovato. Uso il modello base (non finetunato).")
                self.model = self.base_model.to(self.device)

            # Inference mode
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