# Relazione Tecnica: Sistema NLP Avanzato per l'Analisi della Salute Mentale e Sintesi Documentale

**Corso:** Natural Language Processing  
**Progetto:** Faboulus-Interpretr  
**Autore:** Golddiggers  
**Data:** 19 Dicembre 2025

---

## Sommario

Il presente documento illustra la progettazione e l'implementazione di **Faboulus-Interpretr**, un sistema software modulare basato su tecniche avanzate di Natural Language Processing (NLP). Il progetto affronta due domini applicativi distinti ma complementari: l'analisi semantica di testi relativi alla salute mentale (Mental Health Sentiment Analysis) e la sintesi automatica di documentazione tecnica (Abstractive Summarization).

L'architettura proposta integra modelli Transformer allo stato dell'arte, specificamente **XLM-RoBERTa** ottimizzato tramite *Low-Rank Adaptation (LoRA)* per la classificazione, e **IT5 (Italian T5)** per la generazione di riassunti astrattivi. Particolare enfasi è stata posta sull'ingegneria del software, implementando una pipeline di ingestion robusta capace di processare formati eterogenei (PDF, Web, OpenAPI) e un'interfaccia utente interattiva basata su Streamlit. I risultati sperimentali dimostrano un netto vantaggio delle architetture basate su Transformer rispetto ai metodi tradizionali, raggiungendo un'accuratezza del **77.9%** nella classificazione e punteggi **ROUGE-1** del **42%** nella sintesi.

---

## 1. Introduzione

L'elaborazione del linguaggio naturale (NLP) ha subito una rivoluzione paradigmatica con l'avvento dei modelli basati su architettura Transformer, che hanno permesso di superare i limiti delle tecniche statistiche tradizionali nella comprensione del contesto e delle sfumature linguistiche. Questo progetto si colloca in tale scenario, proponendo una soluzione pratica a due problemi di grande rilevanza industriale e sociale.

### 1.1 Obiettivi del Progetto

1.  **Analisi della Salute Mentale:** Sviluppare un classificatore capace di distinguere tra stati emotivi critici in testi non strutturati. La sfida principale risiede nella natura implicita e soggettiva del linguaggio usato per descrivere il disagio psicologico, che richiede una comprensione profonda ("deep understanding") superiore alle semplici keyword.
2.  **Sintesi Documentale Tecnica:** Automatizzare l'estrazione di informazioni chiave da manuali tecnici e specifiche API. In questo dominio, la sfida è opposta: il linguaggio è formale e strutturato, ma i documenti sono spesso lunghi e frammentati, richiedendo strategie avanzate di gestione del contesto (long-context handling).

---

## 2. Stato dell'Arte e Background Tecnico

### 2.1 Dai Modelli Statistici ai Transformer
L'evoluzione dell'NLP ha visto il passaggio da rappresentazioni *Bag-of-Words* (BoW) e TF-IDF, che ignorano l'ordine delle parole, a modelli sequenziali come le RNN e LSTM. Sebbene le LSTM abbiano introdotto la memoria a lungo termine, soffrivano ancora di difficoltà nel parallelismo e nella gestione di dipendenze molto distanti. L'introduzione del meccanismo di *Self-Attention* (Vaswani et al., 2017) ha risolto questi problemi, permettendo ai modelli di "pesare" l'importanza di ogni parola rispetto a tutte le altre nella frase contemporaneamente.

### 2.2 Summarization Astrattiva vs Estrattiva
La sintesi automatica si divide in due approcci:
*   **Estrattiva:** Seleziona e concatena le frasi più importanti del testo originale (es. algoritmo TextRank). Garantisce fedeltà ma spesso manca di coerenza discorsiva.
*   **Astrattiva:** Genera nuovo testo parafrasando i contenuti (es. modelli Seq2Seq come T5, BART). È l'approccio scelto per questo progetto, poiché permette di condensare informazioni tecniche complesse in descrizioni fluide, essenziali per la documentazione API.

### 2.3 Parameter-Efficient Fine-Tuning (PEFT)
L'addestramento completo (Full Fine-Tuning) di Large Language Models (LLM) è computazionalmente oneroso. In questo progetto abbiamo adottato **LoRA (Low-Rank Adaptation)**, una tecnica che congela i pesi del modello pre-addestrato e inietta matrici di rango ridotto addestrabili nei layer di attenzione. Questo riduce i parametri da aggiornare di ordini di grandezza (spesso < 1% del totale), mantenendo prestazioni comparabili al full fine-tuning.

---

## 3. Metodologia e Architettura del Sistema

Il sistema è stato progettato seguendo i principi di modularità e separazione delle responsabilità. Il codice sorgente è organizzato nella directory `src/` e orchestrato da un'applicazione centrale.

### 3.1 Pipeline di Data Ingestion
La qualità dell'output NLP dipende strettamente dalla qualità dell'input. Abbiamo sviluppato un modulo di ingestion (`data_ingestion.py`) polimorfico:

*   **PDF Parsing:** Utilizziamo la libreria `PyMuPDF` (fitz) per estrarre testo preservando la struttura spaziale, fondamentale per distinguere intestazioni e corpo del testo.
*   **Web Scraping Intelligente:** Per le fonti online, utilizziamo `trafilatura`, che impiega euristiche avanzate per estrarre il "main content" di una pagina web, scartando boilerplate, menu di navigazione e footer che introdurrebbero rumore nel riassunto.
*   **OpenAPI/Swagger:** I file JSON/YAML delle API non vengono trattati come testo semplice. Un parser dedicato naviga la struttura ad albero della specifica, estraendo solo i campi semantici (`summary`, `description`, `parameters`) e linearizzandoli in un formato discorsivo ("L'endpoint /users permette di...").

### 3.2 Preprocessing: Chunking Semantico
I modelli Transformer hanno una finestra di contesto limitata (tipicamente 512 o 1024 token). Per processare documenti lunghi, abbiamo implementato un algoritmo di **Recursive Character Text Splitting** (`preprocessing.py`).
A differenza di un taglio arbitrario ogni N caratteri, questo algoritmo tenta di dividere il testo rispettando la gerarchia sintattica: prima sui doppi a capo (paragrafi), poi sui singoli a capo, poi sui punti (fine frase). Questo preserva l'integrità semantica dei frammenti passati al modello.

### 3.3 Modulo di Summarization (Map-Reduce)
Per la sintesi, utilizziamo il modello **IT5 (Italian T5)**, una variante di T5 pre-addestrata su un vasto corpus italiano. Per gestire documenti che superano la finestra di contesto anche dopo il chunking, adottiamo una strategia **Map-Reduce**:
1.  **Map:** Ogni chunk viene riassunto indipendentemente.
2.  **Reduce:** I riassunti parziali vengono concatenati e, se necessario, riassunti nuovamente per produrre l'output finale coerente.

### 3.4 Modulo di Sentiment Analysis (XLM-RoBERTa + LoRA)
Per la classificazione, abbiamo scelto **XLM-RoBERTa**, un modello multilingue robusto. L'adattamento al dominio "Mental Health" è avvenuto tramite LoRA.
Il modulo (`sentiment.py`) include un sistema di caricamento dinamico che seleziona il dispositivo di calcolo ottimale (CUDA su NVIDIA, MPS su Mac Silicon, CPU altrimenti), garantendo la portabilità del codice.

---

## 4. Analisi dei Risultati

### 4.1 Performance di Classificazione
La validazione del modulo di Sentiment Analysis è stata condotta sul dataset `mental_balanced.csv` (~30.000 campioni), confrontando tre diverse architetture.

| Architettura | Feature Extraction | Accuratezza (Test Set) | Note |
| :--- | :--- | :--- | :--- |
| **Naive Bayes** | TF-IDF (n-grams) | 66.2% | Baseline statistica. Veloce ma imprecisa su frasi complesse. |
| **Bi-LSTM** | GloVe Embeddings | 72.6% | Migliora la comprensione sequenziale ma fatica su dipendenze lunghe. |
| **XLM-RoBERTa** | **Transformer + LoRA** | **77.9%** | **State-of-the-Art**. La self-attention cattura efficacemente il contesto emotivo. |

Il vantaggio del +11.7% del Transformer rispetto al baseline statistico conferma che per compiti di comprensione profonda, dove il sentimento è spesso implicito o distribuito su più frasi, le architetture moderne sono imprescindibili.

### 4.2 Qualità della Sintesi
Per la Summarization, in assenza di un "ground truth" perfetto per ogni documento tecnico, abbiamo utilizzato le metriche ROUGE su un set di validazione annotato.
*   **ROUGE-1 (Unigrammi):** **~42%**. Indica che il modello cattura quasi la metà delle parole chiave fondamentali presenti nel riferimento umano.
*   **ROUGE-L (Longest Common Subsequence):** **~38%**. Suggerisce che il modello è in grado di costruire frasi con una struttura sintattica simile a quella umana, mantenendo una buona fluidità.

---

## 5. Sviluppo dell'Interfaccia Utente

L'accessibilità degli strumenti sviluppati è garantita da una Web App realizzata con **Streamlit** (`app.py`). L'interfaccia è progettata per utenza non tecnica:
*   **Upload Drag & Drop:** Supporto per caricamento file PDF e CSV diretti.
*   **Feedback Visivo:** Barre di progresso durante l'inferenza e visualizzazione degli score di confidenza (es. "Sentiment: Anxiety, Confidenza: 92%").
*   **Ispezione:** Possibilità di espandere i dettagli tecnici (es. testo estratto grezzo) per verificare la correttezza dell'ingestion.

---

## 6. Conclusioni

Il progetto **Faboulus-Interpretr** ha dimostrato con successo come le moderne tecnologie NLP possano essere integrate in un'applicazione funzionale e robusta. L'adozione di **LoRA** ha permesso di ottenere prestazioni elevate contenendo i costi computazionali, rendendo il sistema eseguibile anche su hardware consumer. L'approccio ingegneristico alla pipeline dei dati (gestione PDF/OpenAPI) ha risolto le problematiche pratiche spesso trascurate nella ricerca accademica pura.

Gli sviluppi futuri si concentreranno sull'integrazione di tecniche **RAG (Retrieval-Augmented Generation)** per permettere all'utente di "dialogare" con la documentazione tecnica, superando il limite del riassunto statico.

---

## 7. Bibliografia

1.  Vaswani, A., et al. (2017). *Attention Is All You Need*. NIPS.
2.  Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR.
3.  Raffel, C., et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5). JMLR.
4.  Conneau, A., et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale* (XLM-RoBERTa). ACL.
5.  Sarti, G. (2022). *IT5: Large-scale Text-to-text Pretraining for Italian Language Understanding and Generation*.
