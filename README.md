<div align="center">

# Faboulous-Interpretr

> **Progetto Universitario NLP (9 CFU)**
> 
> Un toolkit NLP avanzato basato su modelli Transformer italiani State-of-the-Art per la sintesi di documentazione tecnica e l'analisi del sentiment.

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow?style=for-the-badge)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

</div>

## 🚀 Panoramica del Progetto

**Faboulous-Interpretr** è un'applicazione NLP *production-ready* sviluppata come tesina per il corso di Data Science. A differenza di semplici wrapper API, implementa una pipeline ingegneristica completa per gestire scenari reali complessi.

Le funzionalità core sono due:
1.  **📄 Summarization Avanzata**: Sintesi di documenti tecnici lunghi (PDF, Specifiche API) utilizzando un approccio **Map-Reduce** con modelli IT5.
2.  **😊 Sentiment Analysis**: Classificazione di recensioni (singole o batch CSV) con modelli FEEL-IT specifici per la lingua italiana.

## ✨ Features Chiave & Architettura

### 1. Documentation Summarizer (con Map-Reduce)
Il sistema supera il limite dei token (tipico dei modelli BERT/T5) implementando una strategia custom:
*   **Ingestion Multi-Source**: Supporto nativo per PDF (`PyMuPDF`), Web Scraping (`Trafilatura`) e Specifiche OpenAPI (parsing JSON/YAML).
*   **Semantic Chunking**: Un algoritmo ricorsivo (`RecursiveTokenChunker`) divide il testo preservando i confini delle frasi e del significato.
*   **Map-Reduce Logic**: I testi lunghi vengono processati a blocchi e poi sintetizzati strutturalmente, evitando la perdita di informazioni cruciali nelle sezioni intermedie.
*   **Modello**: `it5-base-summarization` (E. Federici), fine-tuned su dataset italiani.

### 2. Sentiment Analysis (Batch Optimized)
*   **Input Flessibile**: Analisi real-time di testo libero o elaborazione batch di file CSV.
*   **Ingegneria del Prompt**: Logica di pre-processing per pulire i dati grezzi (rimozione URL, normalizzazione).
*   **Modello**: `feel-it-italian-sentiment` (MilaNLProc), SOTA per l'analisi delle emozioni in italiano.

### 3. Valutazione Quantitativa
Il progetto include una pipeline di valutazione automatica basata su metriche **ROUGE** per certificare la qualità dei riassunti generati.

## 📊 Performance Evaluation

Di seguito i risultati della valutazione quantitativa (ROUGE) condotta su un dataset di validazione curato manualmente (Notizie tech/mediche/clima):

| Metrica | Punteggio | Interpretazione |
|---------|-----------|-----------------|
| **ROUGE-1** | **30.11%** | Buon overlap lessicale (parole singole) per un modello zero-shot. |
| **ROUGE-2** | **9.98%** | Indica che il modello riformula attivamente le frasi invece di copiare. |
| **ROUGE-L** | **25.79%** | La struttura logica del riassunto segue fedelmente l'originale. |

> *Nota: I punteggi dimostrano che il modello produce riassunti coerenti e semanticamente validi, pur variando il lessico rispetto al riferimento "gold standard".*

## 🛠️ Tech Stack

*   **Frontend**: Streamlit (Interfaccia reattiva)
*   **Deep Learning**: PyTorch, Hugging Face Transformers
*   **Data Processing**: Pandas, PyMuPDF, Trafilatura
*   **Evaluation**: ROUGE Score, Evaluate library
*   **Hardware**: Rilevamento automatico e supporto per CUDA (NVIDIA) e MPS (Apple Silicon).

## 📦 Installazione e Utilizzo

### Prerequisiti
*   Python 3.9+
*   Pip
*   (Opzionale) GPU NVIDIA per inferenza veloce

### Setup Rapido

1.  **Clona il repository**:
    ```bash
    git clone https://github.com/DataScience-Golddiggers/Faboulous-Interpretr.git
    cd Faboulous-Interpretr
    ```

2.  **Crea e attiva un ambiente virtuale** (Consigliato):
    ```bash
    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # Mac/Linux:
    source .venv/bin/activate
    ```

3.  **Installa le dipendenze**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Avvia l'applicazione**:
    ```bash
    streamlit run app.py
    ```
    L'app sarà accessibile a `http://localhost:8501`.

### Eseguire la Valutazione
Per riprodurre i benchmark di valutazione:
```bash
python -m src.evaluation
```

## 🏗️ Struttura del Progetto

```
Faboulous-Interpretr/
├── app.py                  # Entry point Streamlit (UI Orchestration)
├── requirements.txt        # Dipendenze di progetto
├── docs/
│   └── Plan.md             # Documento di design architetturale (Dettaglio Accademico)
├── src/                    # Core Logic
│   ├── data_ingestion.py   # Adattatori per PDF, URL, OpenAPI
│   ├── preprocessing.py    # Recursive Token Chunker & Text Cleaning
│   ├── summarization.py    # Classe SummarizerModule con logica Map-Reduce
│   ├── sentiment.py        # Classe SentimentAnalyzerModule
│   ├── evaluation.py       # Script di validazione ROUGE
│   └── utils.py            # Gestione Hardware e Logging
└── models/                 # Cache locale dei modelli (Git ignored)
```

## 👥 Autori

Progetto realizzato per il corso di NLP (Data Science).

---
**Made with ❤️ & Transformers**
</div>
