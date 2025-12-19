<div align="center">

# Faboulous-Interpretr

> **Progetto Universitario NLP (9 CFU)**
> 
> Un toolkit NLP avanzato basato su architetture Transformer State-of-the-Art per la sintesi documentale e l'analisi della salute mentale tramite tecniche PEFT (LoRA).

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow?style=for-the-badge)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

</div>

## 🚀 Panoramica del Progetto

**Faboulous-Interpretr** è una piattaforma NLP *production-ready* progettata per affrontare due task complessi di elaborazione del linguaggio naturale: la sintesi di documentazione tecnica estesa e l'identificazione di pattern legati alla salute mentale nei testi.

Il progetto si distingue per l'adozione di tecniche di ottimizzazione avanzate come **Map-Reduce** per la gestione di testi lunghi e **LoRA (Low-Rank Adaptation)** per il fine-tuning efficiente dei modelli.

### Funzionalità Core
1.  **📄 Summarization Strutturata**: Sintesi intelligente di documenti tecnici (PDF, API Specs, Web) mantenendo la coerenza logica tramite chunking ricorsivo.
2.  **🧠 Mental Health Analysis**: Classificazione del testo per l'identificazione di stati emotivi e psicologici (es. *Anxiety*, *Depression*, *Stress*) utilizzando modelli XLM-RoBERTa adattati con LoRA.

## 🏗️ Architettura del Sistema

Il sistema è modulare e progettato per scalare, con una chiara separazione tra ingestione dati, logica di inferenza e interfaccia utente.

### 1. Documentation Summarizer (Map-Reduce)
Per superare i limiti di context window dei Transformer standard, abbiamo implementato una pipeline custom:
*   **Ingestion Agnostica**: Adattatori specifici per PDF (`PyMuPDF`), Web (`Trafilatura`) e file JSON/YAML (OpenAPI).
*   **Recursive Chunking**: Segmentazione semantica del testo che preserva i confini delle frasi per evitare troncamenti brutali.
*   **Map-Reduce Strategy**: Ogni segmento viene riassunto individualmente (Map) e i risultati vengono aggregati strutturalmente (Reduce), garantendo che nessun dettaglio tecnico venga perso.
*   **Backbone**: `it5-base-summarization`, fine-tuned specificamente per la lingua italiana.

### 2. Sentiment & Mental Health Engine (PEFT/LoRA)
Un modulo di classificazione altamente specializzato:
*   **Model Architecture**: `XLM-RoBERTa Base` potenziato con adapter **LoRA**. Questo permette di avere un modello performante con un footprint di memoria ridotto, aggiornando meno dell'1% dei parametri totali durante il training.
*   **Fine-Tuning Pipeline**: Script di training dedicato (`train_sentiment.py`) che gestisce il ciclo di vita del modello, dal preprocessing del dataset al salvataggio degli adapter.
*   **Target Classes**: Configurato per rilevare sfumature complesse (es. *Normal*, *Depression*, *Anxiety*) oltre al classico sentiment positivo/negativo.

## 📂 Struttura della Repository

```text
Faboulous-Interpretr/
├── app.py                  # Entry point Streamlit (UI & Orchestration)
├── requirements.txt        # Dipendenze di produzione
├── data/
│   ├── external/           # Dati da fonti esterne
│   ├── processed/          # Dataset puliti e pronti per il training
│   └── raw/                # Dati grezzi (CSV, PDF, JSON)
├── docs/                   # Documentazione tecnica e accademica
├── models/                 # Model Registry locale (Checkpoint LoRA, Cache HF)
├── notebooks/              # Jupyter Notebooks per EDA e sperimentazione
│   ├── 1_EDA_and_Baseline.ipynb
│   └── sentiment_analysis_nn.ipynb
└── src/                    # Source Code
    ├── data_ingestion.py   # Loader per PDF, URL e OpenAPI
    ├── preprocessing.py    # Text Cleaning e Recursive Token Chunker
    ├── summarization.py    # Logica di inferenza Summarization
    ├── sentiment.py        # Logica di inferenza Sentiment (Caricamento LoRA)
    ├── train_sentiment.py  # Pipeline di training PEFT/LoRA
    ├── evaluation.py       # Script di validazione metriche (ROUGE)
    └── utils.py            # Hardware detection e Logging centralizzato
```

## 🛠️ Tech Stack

*   **Frontend**: Streamlit
*   **Modeling**: PyTorch, Hugging Face Transformers, PEFT (Parameter-Efficient Fine-Tuning)
*   **Data Processing**: Pandas, Scikit-learn
*   **NLP Utils**: PyMuPDF (Fitz), Trafilatura
*   **Hardware Acceleration**: Supporto automatico per CUDA (NVIDIA) e MPS (Apple Silicon).

## 📦 Installazione e Utilizzo

### Prerequisiti
*   Python 3.9+
*   Virtual Environment (consigliato)

### Setup Rapido

1.  **Clona il repository**:
    ```bash
    git clone https://github.com/DataScience-Golddiggers/Faboulous-Interpretr.git
    cd Faboulous-Interpretr
    ```

2.  **Attiva l'ambiente virtuale**:
    ```bash
    # Windows
    python -m venv .venv
    .venv\Scripts\activate
    
    # Unix/MacOS
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Installa le dipendenze**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Avvia la Web App**:
    ```bash
    streamlit run app.py
    ```

### 🧠 Training del Modello (LoRA)

Il progetto include una pipeline completa per il fine-tuning. Per addestrare un nuovo adapter sui propri dati:

```bash
python src/train_sentiment.py \
  --data_path "data/processed/mental_balanced.csv" \
  --text_col "text" \
  --label_col "label" \
  --epochs 5 \
  --batch_size 16 \
  --output_dir "models/my_custom_lora"
```

Il sistema salverà automaticamente gli adapter nella cartella specificata, pronti per essere caricati dal modulo di inferenza.

## 📊 Valutazione

Le performance dei modelli sono monitorate tramite metriche quantitative:
*   **Summarization**: ROUGE-1, ROUGE-2, ROUGE-L.
*   **Classification**: Accuracy, F1-Score (Weighted).

Per eseguire la suite di valutazione:
```bash
python -m src.evaluation
```

---
**Authors**: Data Science Golddiggers Team