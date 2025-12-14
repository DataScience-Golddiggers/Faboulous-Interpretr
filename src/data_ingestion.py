import fitz  # PyMuPDF
import trafilatura
import yaml
import json
import logging
from typing import Optional, Dict, Any

# Configura il logger se non è già stato fatto
logger = logging.getLogger(__name__)

def extract_from_pdf(file_path: str) -> str:
    """
    Estrae testo da un file PDF preservando una struttura leggibile.
    Usa PyMuPDF per velocità ed efficienza.
    """
    text_content = []
    try:
        doc = fitz.open(file_path)
        for page_num, page in enumerate(doc):
            # Estrae testo semplice. Si potrebbe migliorare usando 'blocks' per layout complessi
            text = page.get_text()
            if text.strip():
                text_content.append(text)
        
        doc.close()
        full_text = "\n".join(text_content)
        logger.info(f"Estratti {len(full_text)} caratteri da PDF: {file_path}")
        return full_text
    except Exception as e:
        logger.error(f"Errore nell'estrazione PDF {file_path}: {e}")
        return ""

def extract_from_url(url: str) -> str:
    """
    Scarica e estrae il contenuto principale da una pagina web, rimuovendo boilerplate.
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            logger.error(f"Impossibile scaricare URL: {url}")
            return ""
        
        text = trafilatura.extract(downloaded)
        if text:
            logger.info(f"Estratti {len(text)} caratteri da URL: {url}")
            return text
        else:
            logger.warning(f"Nessun contenuto di testo estratto da: {url}")
            return ""
    except Exception as e:
        logger.error(f"Errore nell'estrazione URL {url}: {e}")
        return ""

def parse_openapi_spec(file_content: str, is_json: bool = False) -> str:
    """
    Converte una specifica OpenAPI (JSON/YAML) in un testo descrittivo discorsivo
    adatto per la summarization.
    """
    try:
        if is_json:
            spec = json.loads(file_content)
        else:
            spec = yaml.safe_load(file_content)
            
        output_text = []
        
        # Info generali
        info = spec.get('info', {})
        title = info.get('title', 'API Document')
        desc = info.get('description', '')
        output_text.append(f"Titolo API: {title}\nDescrizione Generale: {desc}\n")
        
        # Iterazione sui Paths
        paths = spec.get('paths', {})
        for path, methods in paths.items():
            for method, details in methods.items():
                if method in ['get', 'post', 'put', 'delete', 'patch']:
                    summary = details.get('summary', '')
                    description = details.get('description', '')
                    
                    endpoint_desc = f"Endpoint: {method.upper()} {path}"
                    if summary:
                        endpoint_desc += f"\nRiepilogo: {summary}"
                    if description:
                        endpoint_desc += f"\nDettagli: {description}"
                    
                    output_text.append(endpoint_desc)
        
        final_text = "\n\n".join(output_text)
        logger.info(f"Convertita specifica OpenAPI in testo di {len(final_text)} caratteri.")
        return final_text

    except Exception as e:
        logger.error(f"Errore nel parsing OpenAPI: {e}")
        return ""

