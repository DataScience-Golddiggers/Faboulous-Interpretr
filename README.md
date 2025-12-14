<div align="center">

# Faboulous-Interpretr

> An advanced NLP toolkit powered by state-of-the-art Italian Transformer models for technical documentation summarization and sentiment analysis.

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)
![MacOS](https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=apple&logoColor=white)
![Linux](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)


## 🚀 Overview

**Faboulous-Interpretr** is a production-ready NLP application that leverages cutting-edge natural language processing models to provide two core functionalities:

1. **📄 Technical Documentation Summarization** - Automatic summarization using IT5 transformer models
2. **😊 Sentiment Analysis** - Review sentiment classification using FEEL-IT models

The application features a modern Streamlit UI with support for multiple input sources including PDFs, URLs, OpenAPI specifications, and batch CSV processing.

## ✨ Features

### Documentation Summarizer
- **Multiple Input Sources**: 
- Free text input
- PDF document parsing
- Web scraping from URLs
- OpenAPI specification (JSON/YAML) parsing
- **IT5-based Summarization**:  State-of-the-art Italian text summarization
- **Export Capabilities**: Download summaries in text format

### Sentiment Analysis
- **Single Review Analysis**: Real-time sentiment classification
- **Batch Processing**: CSV upload for bulk analysis
- **FEEL-IT Integration**: Italian-optimized sentiment detection
- **Visualization**: Interactive charts and statistics
- **Export Results**: Download analyzed datasets with sentiment labels

## 🛠️ Tech Stack

- **Framework**: Streamlit
- **NLP Models**:  Hugging Face Transformers (IT5, FEEL-IT)
- **Data Processing**: Pandas, PyMuPDF
- **Visualization**: Plotly Express
- **Hardware Acceleration**: CUDA, Apple MPS, CPU fallback

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/DataScience-Golddiggers/Faboulous-Interpretr.git
cd Faboulous-Interpretr
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

The application will launch in your default browser at `http://localhost:8501`

## 🏗️ Project Structure

```
Faboulous-Interpretr/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── src/                    # Source modules
│   ├── data_ingestion.py  # PDF/URL/API parsing utilities
│   ├── summarization.py   # Summarizer module wrapper
│   ├── sentiment. py       # Sentiment analyzer wrapper
│   └── utils.py           # Logging and device detection
├── models/                 # Model cache directory
└── docs/                   # Documentation files
```

## 🎯 Usage

### Documentation Summarization

1. Navigate to **📄 Doc Summarizer** from the sidebar
2. Select your input source:
- **Free Text**:  Paste text directly
- **PDF**: Upload a PDF file
- **URL**:  Provide a web page URL
- **OpenAPI**:  Upload API specification
3. Click **Generate Summary**
4. Download or copy the generated summary

### Sentiment Analysis

**Single Analysis:**
1. Navigate to **😊 Sentiment Analysis**
2. Select **Single Analysis**
3. Enter your review text
4. Click **Analyze** to see sentiment and confidence score

**Batch Processing:**
1. Select **Upload CSV Batch**
2. Upload a CSV file with a column named `text`, `recensione`, `body`, or `content`
3. Click **Analyze Dataset**
4. View distribution charts and download results

## 🔧 Configuration

The application automatically detects your hardware setup:
- **NVIDIA GPU**: CUDA acceleration
- **Apple Silicon**: MPS acceleration
- **CPU**: Standard processing

## 📊 Models

- **Summarization**: IT5 (Italian T5) - Fine-tuned for Italian technical text
- **Sentiment Analysis**:  FEEL-IT - Italian emotion and sentiment classifier

Models are automatically downloaded on first use and cached locally in the `models/` directory.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

**DataScience-Golddiggers** - [GitHub Organization](https://github.com/DataScience-Golddiggers)

## 🙏 Acknowledgments

- Hugging Face for providing excellent NLP models
- Streamlit for the intuitive web framework
- The open-source community for various libraries and tools

## 📧 Support

For issues, questions, or contributions, please open an issue on the [GitHub repository](https://github.com/DataScience-Golddiggers/Faboulous-Interpretr/issues).

---

**Made with ❤️ by DataScience-Golddiggers**
</div>