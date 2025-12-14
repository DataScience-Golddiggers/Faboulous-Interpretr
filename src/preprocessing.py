import re
import logging
from typing import List

logger = logging.getLogger(__name__)

class RecursiveTokenChunker:
    """
    Divide il testo in chunk rispettando un limite massimo di caratteri (proxy per token),
    cercando di tagliare su separatori naturali (paragrafi, frasi, parole).
    
    È una versione leggera "pure Python" ispirata a LangChain.
    """
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Separatori in ordine di priorità: Doppio a capo (paragrafo), A capo, Punto, Spazio
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        final_chunks = []
        if not text:
            return final_chunks
            
        # Normalizza un po' il testo prima di splittare
        text = re.sub(r'\r\n', '\n', text)
        
        self._split_text_recursive(text, self.separators, final_chunks)
        
        logger.info(f"Testo diviso in {len(final_chunks)} chunk (Dimensione max: {self.chunk_size}).")
        return final_chunks

    def _split_text_recursive(self, text: str, separators: List[str], final_chunks: List[str]):
        """
        Funzione ricorsiva interna per il splitting.
        """
        final_chunks_len = len(final_chunks)
        
        # Se il testo è già abbastanza piccolo, aggiungilo (se non è vuoto)
        if len(text) <= self.chunk_size:
            final_chunks.append(text)
            return

        # Se non ci sono più separatori, siamo costretti a tagliare brutalmente
        if not separators:
            # Taglio brutale ogni chunk_size
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                final_chunks.append(text[i : i + self.chunk_size])
            return

        # Prendi il separatore corrente
        separator = separators[0]
        next_separators = separators[1:]
        
        # Se il separatore non c'è, passa al prossimo
        if separator and separator not in text:
            self._split_text_recursive(text, next_separators, final_chunks)
            return

        # Split del testo
        # Se il separatore è stringa vuota (ultimo caso), dividiamo per caratteri
        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)

        # Ora ricomponiamo i pezzi in chunk validi
        current_chunk = []
        current_length = 0
        
        for split in splits:
            # Riaggiungiamo il separatore se non è vuoto (per mantenere la formattazione)
            split_len = len(split) + (len(separator) if separator else 0)
            
            if current_length + split_len > self.chunk_size:
                # Il chunk corrente è pieno, processiamolo
                # Se abbiamo accumulato qualcosa, lo mandiamo giù ricorsivamente 
                # (per gestire casi dove anche un singolo split è > chunk_size)
                doc_to_process = (separator if separator else "").join(current_chunk)
                if doc_to_process:
                    # Se il doc è ancora troppo grande (perché un singolo split era enorme),
                    # la ricorsione con il prossimo separatore lo gestirà.
                    # Se è giusto, verrà aggiunto subito.
                    if len(doc_to_process) > self.chunk_size:
                         self._split_text_recursive(doc_to_process, next_separators, final_chunks)
                    else:
                        final_chunks.append(doc_to_process)
                
                # Reset chunk corrente, gestendo overlap (semplificato qui: ripartiamo dal pezzo corrente)
                current_chunk = [split]
                current_length = split_len
            else:
                current_chunk.append(split)
                current_length += split_len
        
        # Aggiungi l'ultimo pezzo rimasto
        if current_chunk:
            doc_to_process = (separator if separator else "").join(current_chunk)
            if len(doc_to_process) > self.chunk_size:
                 self._split_text_recursive(doc_to_process, next_separators, final_chunks)
            else:
                final_chunks.append(doc_to_process)

def clean_text(text: str) -> str:
    """
    Pulizia base del testo per rimuovere rumore evidente.
    """
    # Rimuove spazi multipli
    text = re.sub(r'\s+', ' ', text).strip()
    return text
