# Relazione Tecnica: Sistema NLP Avanzato per l'Analisi della Salute Mentale e Sintesi Documentale

**Progetto:** Faboulus-Interpretr  
**Ambito:** Natural Language Processing & Software Engineering  
**Autore:** Golddiggers

---

## 1. Executive Summary
Il progetto **Faboulus-Interpretr** è un sistema integrato per l'elaborazione del linguaggio naturale che affronta due sfide critiche: la classificazione semantica di testi legati alla salute mentale e la sintesi automatica di documentazione tecnica eterogenea. 

Il sistema si distingue per un'architettura **model-agnostic** e **scalabile**, che integra:
1.  Un motore di **Sentiment Analysis** basato su Transformer (XLM-RoBERTa) ottimizzato tramite tecniche di *Parameter-Efficient Fine-Tuning* (LoRA).
2.  Un modulo di **Abstractive Summarization** (IT5) con gestione intelligente di documenti lunghi tramite logica Map-Reduce.
3.  Una **pipeline di Ingestion** capace di processare PDF, URL e specifiche OpenAPI (Swagger).
4.  Un'interfaccia grafica interattiva sviluppata in **Streamlit**.

---

## 2. Architettura del Sistema (Core Modules)

Il cuore del progetto risiede nella cartella `src/`, dove la logica è stata modularizzata per garantire manutenibilità e testabilità.

### 2.1 Ingestion & Parsing (`data_ingestion.py`)
Il sistema non accetta solo testo semplice, ma gestisce la complessità del mondo reale:
*   **PDF Extraction**: Utilizza `PyMuPDF` (fitz) per un'estrazione veloce e strutturata.
*   **Web Scraping**: Implementa `trafilatura` per isolare il contenuto principale degli articoli web, eliminando automaticamente menu, pubblicità e boilerplate.
*   **OpenAPI Parser**: Una funzione dedicata converte file JSON/YAML tecnici in un "testo discorsivo" leggibile dal modello di sintesi, trasformando endpoint e parametri in descrizioni naturali.

### 2.2 Preprocessing & Chunking Ricorsivo (`preprocessing.py`)
Uno dei problemi maggiori dei Transformer è il limite dei token (es. 512). Abbiamo implementato un **`RecursiveTokenChunker`**:
*   A differenza di uno split brutale, questo algoritmo divide il testo cercando "separatori naturali" in ordine di priorità: `\n\n` (paragrafi), `\n` (linee), `. ` (frasi) e infine spazi.
*   Questo garantisce che il modello riceva chunk di testo semanticamente coerenti, migliorando drasticamente la qualità del riassunto finale.

### 2.3 Modulo di Summarization (`summarization.py`)
Il modulo `SummarizerModule` utilizza il modello **IT5** (Italian T5). La vera innovazione è la gestione dei testi lunghi:
*   **Strategia Map-Reduce**: Se un documento è troppo lungo, viene diviso in sezioni. Ogni sezione viene riassunta singolarmente e i "micro-riassunti" vengono poi aggregati in un output strutturato per punti, evitando la perdita di dettagli critici tipica della sintesi a passo singolo.

### 2.4 Modulo di Sentiment con LoRA (`sentiment.py`)
Il modulo `SentimentAnalyzerModule` carica dinamicamente un adapter **LoRA** (Low-Rank Adaptation).
*   **Efficienza**: Invece di caricare un modello da 1GB ogni volta, il sistema carica il modello base e applica un "layer" di pesi leggero (pochi MB) specifico per il task della salute mentale.
*   **Inferenza Intelligente**: Il codice rileva automaticamente se è disponibile una GPU (CUDA) o un chip Apple (MPS) tramite `utils.py`, ottimizzando i tempi di risposta.

---

## 3. Risultati Sperimentali e Validazione

### 3.1 Classificazione (Sentiment Analysis)
Abbiamo confrontato tre architetture sul dataset `mental_balanced.csv` (~30.000 campioni):

| Modello | Metodo | Accuracy | Vantaggio Chiave |
| :--- | :--- | :--- | :--- |
| **Naive Bayes** | Statistico / TF-IDF | 66.2% | Estrema velocità, ma ignora il contesto. |
| **Bi-LSTM** | Deep Learning / GloVe | 72.6% | Cattura la sequenzialità del testo. |
| **XLM-RoBERTa** | **Transformer + LoRA** | **77.9%** | Comprensione semantica profonda (SOTA). |

**Osservazione** Il passaggio al Transformer non è solo "moda". La differenza del +11.7% rispetto al baseline dimostra che la **Self-Attention** è fondamentale per decodificare stati d'animo complessi espressi in forma scritta.

### 3.2 Valutazione della Sintesi (`evaluation.py`)
La qualità dei riassunti è stata validata quantitativamente tramite metriche **ROUGE**:
*   **ROUGE-1**: ~42% (ottima copertura dei concetti principali).
*   **ROUGE-L**: ~38% (buona fluidità e struttura della frase).

---

## 4. Interfaccia Utente (UI) - Streamlit

L'applicazione (`app.py`) funge da pannello di controllo per l'intero ecosistema. È strutturata in due aree principali:

1.  **Dashboard di Analisi Sentiment**:
    *   Permette l'input manuale o il caricamento di file batch.
    *   Visualizza non solo la classe predetta (es. "Serious"), ma anche la **confidence score** (probabilità), dando trasparenza all'utente sull'affidabilità della previsione.
2.  **Summarizer Tecnico**:
    *   Accetta file PDF o URL.
    *   Mostra il testo estratto "grezzo" in un'area espandibile per verifica e il riassunto finale in evidenza.
    *   Gestisce l'elaborazione asincrona per non bloccare l'interfaccia durante l'uso di modelli pesanti.

---

## 5. Conclusioni e Sviluppi Futuri

Il progetto dimostra che l'integrazione di tecniche di **Deep Learning d'avanguardia (LoRA)** con pratiche di **Software Engineering solide (Modularità, Ingestion strutturata)** permette di creare strumenti NLP realmente utili.

Sviluppi futuri potrebbero includere:
*   L'integrazione di **LLM Generativi (es. Llama 3)** tramite RAG (Retrieval-Augmented Generation) per interrogare la documentazione tecnica invece di riassumerla soltanto.
*   Espansione della Sentiment Analysis verso il rilevamento delle emozioni specifiche (Joy, Anger, Fear) per un supporto psicologico più granulare.