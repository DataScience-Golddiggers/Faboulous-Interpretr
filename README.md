<div align="center">

# Faboulous-Interpretr
> **University Course Project - Data Science**
> 
> An advanced NLP toolkit based on State-of-the-Art Transformer architectures for document summarization and mental health analysis using PEFT (LoRA) techniques.

![ProjectCover](docs/public/rdm1.png)

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E.svg?style=for-the-badge&logo=Hugging-Face&logoColor=black)
![nvidia](https://img.shields.io/badge/cuda-76B900.svg?style=for-the-badge&logo=NVIDIA&logoColor=white)
![amd](https://img.shields.io/badge/Radeon-ED1C24.svg?style=for-the-badge&logo=AMD&logoColor=white)
![Metal](https://img.shields.io/badge/MPS-000000.svg?style=for-the-badge&logo=Apple&logoColor=white)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

</div>

## 🚀 Project Overview

**Faboulous-Interpretr** is a *production-ready* NLP platform designed to address two complex natural language processing tasks: the summarization of extensive technical documentation and the identification of mental health-related patterns in text.

The project stands out for its adoption of advanced optimization techniques such as **Map-Reduce** for managing long texts and **LoRA (Low-Rank Adaptation)** for efficient model fine-tuning.

### Core Features
1.  **📄 Structured Summarization**: Intelligent synthesis of technical documents (PDF, API Specs, Web) while maintaining logical coherence through recursive chunking.
2.  **🧠 Mental Health Analysis**: Text classification for identifying emotional and psychological states (e.g., *Anxiety*, *Depression*, *Stress*) using XLM-RoBERTa models adapted with LoRA.

## 🏗️ System Architecture

The system is modular and designed to scale, with a clear separation between data ingestion, inference logic, and user interface.

### 1. Documentation Summarizer (Map-Reduce)
To overcome the context window limits of standard Transformers, we implemented a custom pipeline:
*   **Agnostic Ingestion**: Specific adapters for PDF (`PyMuPDF`), Web (`Trafilatura`), and JSON/YAML files (OpenAPI).
*   **Recursive Chunking**: Semantic text segmentation that preserves sentence boundaries to avoid brutal truncation.
*   **Map-Reduce Strategy**: Each segment is summarized individually (Map) and results are structurally aggregated (Reduce), ensuring no technical detail is lost.
*   **Backbone**: `it5-base-summarization`, fine-tuned specifically for the Italian language.

### 2. Sentiment & Mental Health Engine (PEFT/LoRA)
A highly specialized classification module:
*   **Model Architecture**: `XLM-RoBERTa Base` enhanced with **LoRA** adapters. This allows for a high-performance model with a reduced memory footprint, updating less than 1% of total parameters during training.
*   **Fine-Tuning Pipeline**: Dedicated training script (`train_sentiment.py`) managing the model lifecycle, from dataset preprocessing to adapter saving.
*   **Target Classes**: Configured to detect complex nuances (e.g., *Normal*, *Depression*, *Anxiety*) beyond classic positive/negative sentiment.

## 📂 Repository Structure

```text
Faboulous-Interpretr/
├── app.py                  # Streamlit Entry point (UI & Orchestration)
├── requirements.txt        # Production dependencies
├── data/
│   ├── external/           # Data from external sources
│   ├── processed/          # Cleaned datasets ready for training
│   └── raw/                # Raw data (CSV, PDF, JSON)
├── docs/                   # Technical and academic documentation
├── models/                 # Local Model Registry (LoRA Checkpoints, HF Cache)
├── notebooks/              # Jupyter Notebooks for EDA and experimentation
│   ├── 1_EDA_and_Baseline.ipynb
│   └── sentiment_analysis_nn.ipynb
└── src/                    # Source Code
    ├── data_ingestion.py   # Loaders for PDF, URL, and OpenAPI
    ├── preprocessing.py    # Text Cleaning and Recursive Token Chunker
    ├── summarization.py    # Summarization inference logic
    ├── sentiment.py        # Sentiment inference logic (LoRA Loading)
    ├── train_sentiment.py  # PEFT/LoRA training pipeline
    ├── evaluation.py       # Metrics validation script (ROUGE)
    └── utils.py            # Hardware detection and centralized Logging
```

## 🛠️ Tech Stack

*   **Frontend**: Streamlit
*   **Modeling**: PyTorch, Hugging Face Transformers, PEFT (Parameter-Efficient Fine-Tuning)
*   **Data Processing**: Pandas, Scikit-learn
*   **NLP Utils**: PyMuPDF (Fitz), Trafilatura
*   **Hardware Acceleration**: Automatic support for CUDA (NVIDIA) and MPS (Apple Silicon).

## 📦 Installation and Usage

### Prerequisites
*   Python 3.9+
*   Virtual Environment (recommended)

### Quick Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/DataScience-Golddiggers/Faboulous-Interpretr.git
    cd Faboulous-Interpretr
    ```

2.  **Activate the virtual environment**:
    ```bash
    # Windows
    python -m venv .venv
    .venv\Scripts\activate
    
    # Unix/MacOS
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Start the Web App**:
    ```bash
    streamlit run app.py
    ```

### 🧠 Model Training (LoRA)

The project includes a complete pipeline for fine-tuning. To train a new adapter on your own data:

```bash
python src/train_sentiment.py \
  --data_path "data/processed/mental_balanced.csv" \
  --text_col "text" \
  --label_col "label" \
  --epochs 5 \
  --batch_size 16 \
  --output_dir "models/my_custom_lora"
```

The system will automatically save the adapters in the specified folder, ready to be loaded by the inference module.

## 📊 Evaluation

Model performances are monitored via quantitative metrics:
*   **Summarization**: ROUGE-1, ROUGE-2, ROUGE-L.
*   **Classification**: Accuracy, F1-Score (Weighted).

To run the evaluation suite:
```bash
python -m src.evaluation
```

---
**Authors**: Data Science Golddiggers Team
