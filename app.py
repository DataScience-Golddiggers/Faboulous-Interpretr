import os
import streamlit as st
import pandas as pd
import plotly.express as px
import time
from src.data_ingestion import extract_from_pdf, extract_from_url, parse_openapi_spec
from src.summarization import SummarizerModule
from src.sentiment import SentimentAnalyzerModule
from src.utils import setup_logging, get_device as device


# Configurazione Pagina
st.set_page_config(
    page_title="AI NLP Toolkit",
    page_icon="🤖",
    layout="wide"
)

setup_logging()

# --- Funzioni Caricamento Modelli (Cachate) ---
@st.cache_resource
def load_summarizer():
    return SummarizerModule()

@st.cache_resource
def load_sentiment_analyzer():
    return SentimentAnalyzerModule()

# --- Sidebar Navigazione ---
st.sidebar.title("Navigazione")
app_mode = st.sidebar.radio("Scegli Modulo:", ["🏠 Home", "📄 Doc Summarizer", "😊 Sentiment Analysis"])

# --- Pagina Home ---
if app_mode == "🏠 Home":
    st.title("Sistema NLP Avanzato")
    st.markdown("""
    Benvenuto nella suite di analisi testuale AI.
    
    Questo tool offre due funzionalità principali basate su modelli Transformer State-of-the-Art per l'italiano:
    
    1.  **Technical Summarization:** Sintesi automatica di documentazione tecnica (PDF, URL, OpenAPI) usando **IT5**.
    2.  **Review Sentiment Analysis:** Analisi del sentiment su recensioni utente usando **FEEL-IT**.
    
    👈 **Seleziona un modulo dalla barra laterale per iniziare.**
    """)
    
    lol = ""

    if(os.name == "nt"):
        lol = "Windows"
    elif(os.name == "posix"):
        lol = "Linux/Mac"
    else:
        lol = "Che cinesata è fra?"

    hw = ""
    if str(device()) == "cuda":
        hw = "CUDA (NVIDIA)"
    elif str(device()) == "mps":
        hw = "Apple MPS"
    else:
        hw = "CPU"
    

    st.info(f"Sistema in esecuzione su: {lol} | Accelerazione hardware rilevata: {hw}.")

# --- Pagina Summarizer ---
elif app_mode == "📄 Doc Summarizer":
    st.title("Sintesi Documentazione Tecnica")
    st.write("Carica un documento o incolla un URL per ottenere un riassunto strutturato.")
    
    # Input
    source_type = st.radio("Fonte Dati:", ["Testo Libero", "PDF", "URL Web", "OpenAPI Spec (JSON/YAML)"])
    
    input_text = ""
    
    if source_type == "Testo Libero":
        input_text = st.text_area("Incolla qui il testo:", height=300)
        
    elif source_type == "PDF":
        uploaded_file = st.file_uploader("Carica file PDF", type=["pdf"])
        if uploaded_file:
            with st.spinner("Estrazione testo da PDF..."):
                # Salvataggio temporaneo per fitz
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                input_text = extract_from_pdf("temp.pdf")
                st.success(f"Testo estratto: {len(input_text)} caratteri")
                
    elif source_type == "URL Web":
        url = st.text_input("Inserisci URL (es. ReadTheDocs page):")
        if url:
            with st.spinner("Scraping contenuto..."):
                input_text = extract_from_url(url)
                if input_text:
                    st.success(f"Contenuto scaricato: {len(input_text)} caratteri")
                else:
                    st.error("Impossibile scaricare il contenuto. Controlla l'URL.")

    elif source_type == "OpenAPI Spec (JSON/YAML)":
        uploaded_file = st.file_uploader("Carica specifica API", type=["json", "yaml", "yml"])
        if uploaded_file:
            content = uploaded_file.read().decode("utf-8")
            is_json = uploaded_file.name.endswith(".json")
            with st.spinner("Parsing specifica API..."):
                input_text = parse_openapi_spec(content, is_json=is_json)
                st.success("Specifica convertita in testo descrittivo.")

    # Processamento
    if st.button("Genera Riassunto"):
        if not input_text:
            st.warning("Per favore fornisci un testo o un file valido.")
        else:
            try:
                with st.spinner("Caricamento modello e generazione riassunto (può richiedere tempo)..."):
                    summarizer = load_summarizer()
                    summary = summarizer.summarize(input_text)
                
                st.subheader("Risultato:")
                st.markdown(f"> {summary}")
                
                # Opzione Download
                st.download_button("Scarica Riassunto", summary, file_name="riassunto.txt")
                
                # Debug: Mostra testo originale
                with st.expander("Vedi testo originale estratto"):
                    st.text(input_text)
                    
            except Exception as e:
                st.error(f"Si è verificato un errore: {e}")

# --- Pagina Sentiment ---
elif app_mode == "😊 Sentiment Analysis":
    st.title("Analisi Sentiment Recensioni")
    
    input_method = st.radio("Metodo Input:", ["Analisi Singola", "Upload CSV Batch"])
    
    if input_method == "Analisi Singola":
        text = st.text_area("Scrivi una recensione:", "Il servizio è stato ottimo, ma il cibo era freddo.")
        if st.button("Analizza"):
            with st.spinner("Analisi in corso..."):
                analyzer = load_sentiment_analyzer()
                result = analyzer.analyze(text)
                
            label = result['label']
            score = result['score']
            
            # Colore badge per le 7 classi mental health
            color_map = {
                "normal": "green",
                "anxiety": "orange",
                "stress": "orange", 
                "depression": "red",
                "bipolar": "red",
                "personality disorder": "red",
                "suicidal": "red"
            }
            color = color_map.get(label, "gray")
            st.markdown(f"### Stato Mentale: :{color}[{label.upper()}]")
            st.progress(score, text=f"Confidenza: {score:.2f}")

    elif input_method == "Upload CSV Batch":
        st.info("Carica un file CSV con una colonna chiamata 'text' o 'recensione'.")
        uploaded_file = st.file_uploader("Carica CSV", type=["csv"])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("Anteprima Dati:", df.head())
            
            # Cerca colonna testo
            cols = df.columns
            text_col = next((c for c in cols if c.lower() in ['text', 'recensione', 'body', 'content']), None)
            
            if not text_col:
                text_col = st.selectbox("Seleziona la colonna contenente il testo:", cols)
            
            if st.button("Analizza Dataset"):
                analyzer = load_sentiment_analyzer()
                
                # Barra di progresso
                progress_bar = st.progress(0)
                results = []
                texts = df[text_col].tolist()
                
                for i, text in enumerate(texts):
                    # Gestione valori nulli
                    if pd.isna(text):
                        results.append(None)
                    else:
                        res = analyzer.analyze(str(text))
                        results.append(res['label'])
                    
                    if i % 10 == 0:
                        progress_bar.progress((i + 1) / len(texts))
                
                progress_bar.progress(1.0)
                
                df['sentiment'] = results
                
                st.success("Analisi completata!")
                
                # Visualizzazione Grafici
                st.subheader("Distribuzione Sentiment")
                
                # Pie Chart
                fig = px.pie(df, names='sentiment', title='Distribuzione Classi')
                st.plotly_chart(fig)
                
                # Tabella Risultati
                st.dataframe(df)
                
                # Download
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Scarica CSV con Sentiment", csv, "reviews_analyzed.csv", "text/csv")

