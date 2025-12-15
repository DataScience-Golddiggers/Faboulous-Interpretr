import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import logging
import os
from src.utils import get_device
from src.preprocessing import RecursiveTokenChunker

logger = logging.getLogger(__name__)

class SummarizerModule:
    def __init__(self, model_name: str = "efederici/it5-base-summarization"):
        self.device = get_device()
        
        # Definisci il percorso per la cache locale dei modelli
        self.models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
        
        logger.info(f"Caricamento modello Summarization su {self.device}...")
        logger.info(f"Cache modelli locale: {self.models_dir}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.models_dir)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=self.models_dir).to(self.device)
            
            # Crea pipeline
            self.summarizer = pipeline(
                "summarization", 
                model=self.model, 
                tokenizer=self.tokenizer, 
                device=self.device
            )
            logger.info("Modello Summarization caricato con successo.")
            
        except Exception as e:
            logger.error(f"Errore caricamento modello {model_name}: {e}")
            raise e

        # Inizializza il chunker
        # Aumentiamo leggermente la dimensione del chunk per dare più contesto
        self.chunker = RecursiveTokenChunker(chunk_size=3000, chunk_overlap=300)

    def summarize(self, text: str) -> str:
        """
        Esegue la summarization.
        Se il testo è lungo, restituisce un riassunto strutturato per punti (sezioni),
        evitando di comprimere eccessivamente l'informazione.
        """
        if not text.strip():
            return "Nessun testo fornito."

        # 1. Chunking
        chunks = self.chunker.split_text(text)
        logger.info(f"Avvio summarization su {len(chunks)} chunk.")
        
        # Caso semplice: testo breve
        if len(chunks) == 1:
            return self._summarize_chunk(chunks[0])
        
        # Caso complesso: testo lungo -> Lista puntata delle sezioni
        partial_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}...")
            # Aggiungiamo un indicatore di sezione per chiarezza
            section_summary = self._summarize_chunk(chunk)
            partial_summaries.append(f"**Sezione {i+1}:** {section_summary}")
            
        # Uniamo i riassunti con doppio a capo invece di ri-riassumerli
        # Questo mantiene il dettaglio di ogni parte.
        final_output = "Il documento è stato analizzato in più parti per mantenere i dettagli:\n\n" + "\n\n".join(partial_summaries)
        
        return final_output

    def _summarize_chunk(self, text: str) -> str:
        try:
            # Logica dinamica più generosa
            input_len = len(text.split())
            
            # Parametri ottimizzati per maggiore verbosità
            # Minimo 50 parole, massimo 50% dell'input o 512 token
            min_len = max(50, int(input_len * 0.15))
            max_len = min(int(input_len * 0.6), 512) 
            
            # Correzione se min > max
            if min_len >= max_len:
                min_len = int(max_len * 0.8)

            output = self.summarizer(
                text, 
                max_length=max_len, 
                min_length=min_len,
                length_penalty=2.0,       # Incoraggia output più lunghi
                no_repeat_ngram_size=3,   # Evita ripetizioni a macchinetta
                num_beams=4,              # Migliore qualità di ricerca
                early_stopping=True,
                truncation=True
            )
            return output[0]['summary_text']
        except Exception as e:
            logger.error(f"Errore durante summarization chunk: {e}")
            return "Errore nell'elaborazione di questa sezione."
