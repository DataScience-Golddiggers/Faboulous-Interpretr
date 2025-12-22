# **Progettazione e Implementazione di un Sistema NLP Avanzato per la Sintesi di Documentazione Tecnica e l'Analisi del Sentiment**

## **1\. Introduzione e Contesto Strategico**

Nel panorama contemporaneo della scienza dei dati e dell'intelligenza artificiale, la capacità di elaborare e sintetizzare grandi volumi di informazioni testuali non strutturate rappresenta una delle competenze più critiche e ricercate. Il progetto proposto, finalizzato alla realizzazione di un sistema duale per la sintesi automatica di documentazione tecnica (API) e l'analisi del sentiment su recensioni utente, si colloca all'intersezione tra l'Information Retrieval (IR), il Natural Language Processing (NLP) avanzato e l'ingegneria del software. La natura eterogenea dei dati di input—da una parte testi tecnici altamente strutturati e densi di gergo, dall'altra espressioni libere, colloquiali e soggettive—richiede un approccio architetturale sofisticato che trascenda l'applicazione ingenua di librerie standard.

L'obiettivo di questo rapporto è fornire un piano di implementazione esaustivo, tecnicamente rigoroso e scientificamente fondato, che guidi lo sviluppo di tale sistema dalla fase di acquisizione dei dati fino al deployment di un'interfaccia utente interattiva. Analizzeremo le sfide intrinseche alla gestione di formati documentali complessi come PDF e specifiche OpenAPI, discuteremo le strategie di chunking semantico necessarie per processare testi lunghi con modelli Transformer come T5 e BART, e approfondiremo le tecniche di fine-tuning per l'adattamento di modelli BERT alla lingua italiana e al dominio delle recensioni.

### **1.1 Il Paradigma della Pipeline NLP nel Contesto Industriale**

Lo sviluppo di applicazioni NLP robuste non è mai un processo monolitico, bensì un ciclo iterativo strutturato in fasi distinte ma interdipendenti: acquisizione, pulizia, pre-processing, modellazione, valutazione e monitoraggio.1 Nel contesto specifico di questo progetto, la pipeline si biforca in due flussi paralleli che condividono le fondamenta infrastrutturali ma divergono nelle metodologie di trattamento del segnale linguistico.

Per il **flusso di sintesi documentale**, la sfida primaria risiede nella preservazione della fedeltà informativa. La documentazione API non ammette "allucinazioni"; un riassunto che inventa un parametro di funzione o inverte una logica di autenticazione è inutile, se non dannoso. Qui, l'enfasi è posta su tecniche di *retrieval* accurate, segmentazione strutturale e modelli *sequence-to-sequence* (Seq2Seq) capaci di astrazione controllata.2

Per il **flusso di analisi del sentiment**, la priorità si sposta sulla comprensione delle sfumature soggettive, del sarcasmo e delle polarità implicite in testi spesso sgrammaticati o ricchi di *noise* (emoji, hashtag). La sfida tecnica qui è la gestione dello sbilanciamento delle classi (class imbalance) e l'adattamento al dominio specifico, poiché il linguaggio usato per recensire un ristorante differisce sostanzialmente da quello usato per un software bancario.3

### **1.2 Architettura Logica del Sistema**

Il sistema proposto sarà strutturato secondo un'architettura a microservizi logici, orchestrati da un'applicazione centrale.

1. **Modulo di Ingestion:** Responsabile dell'acquisizione e normalizzazione dei dati da fonti eterogenee (Web, PDF, File System).  
2. **Modulo di Pre-processing:** Dedicato alla pulizia, tokenizzazione e segmentazione intelligente dei testi.  
3. **Core NLP Engine:** Il cuore computazionale, ospitante i modelli Transformer per la sintesi (e.g., IT5/T5) e la classificazione (e.g., UmBERTo/BERT).  
4. **Modulo di Valutazione:** Un framework per il monitoraggio delle performance attraverso metriche quantitative (ROUGE, F1-Score) e qualitative.  
5. **Interfaccia Utente (UI):** Un front-end interattivo basato su Streamlit per l'interazione con l'utente finale.

Nelle sezioni successive, dissezioneremo ciascuno di questi componenti, valutando le opzioni tecnologiche disponibili, i trade-off ingegneristici e le best practices emerse dalla ricerca accademica e industriale.

## **2\. Modulo di Acquisizione ed Elaborazione della Documentazione API**

L'acquisizione di documentazione tecnica è un compito ingannevolmente complesso. A differenza dello scraping di siti di notizie o blog, la documentazione API è spesso dispersa su più pagine, annidata in strutture gerarchiche profonde o incapsulata in formati non testuali come il PDF. Inoltre, la presenza massiccia di blocchi di codice, tabelle di parametri e specifiche JSON/YAML richiede strategie di parsing dedicate per distinguere il contenuto discorsivo (utile per il riassunto) dai dettagli sintattici di basso livello.

### **2.1 Strategie di Ingestion per Formati Eterogenei**

Il sistema deve supportare input multimodali: URL diretti a documentazione online, file PDF e file di testo strutturato (Markdown, TXT). Ogni formato presenta peculiarità che influenzano la qualità del testo estratto.

#### **2.1.1 Estrazione Avanzata da PDF (Portable Document Format)**

Il formato PDF è stato progettato per la fedeltà visiva, non per la struttura semantica. Il testo in un PDF è spesso una collezione di istruzioni di posizionamento caratteri, prive di nozioni come "paragrafo" o "tabella". L'estrazione efficace richiede strumenti capaci di ricostruire questa struttura latente.

Analisi comparativa delle librerie Python per PDF:

| Libreria | Caratteristiche Principali | Pro | Contro | Utilizzo nel Progetto |
| :---- | :---- | :---- | :---- | :---- |
| **PyMuPDF (fitz)** | Wrapper per MuPDF, rendering grafico e testuale | Velocità estrema, preservazione layout, estrazione metadati font 5 | API complessa per principianti | **Scelta Primaria** per PDF nativi digitali |
| **PyPDF2 / pypdf** | Pure Python, manipolazione pagine | Leggero, nessuna dipendenza esterna, facile installazione 5 | Estrazione testo meno accurata, difficoltà con layout complessi | Backup per operazioni di split/merge |
| **PDFMiner.six** | Focus sull'analisi del layout | Eccellente ricostruzione della struttura logica, parametri configurabili 7 | Lento su documenti grandi, verboso | Analisi profonda se PyMuPDF fallisce |
| **OCR (Tesseract)** | Riconoscimento ottico caratteri | Indispensabile per PDF scansionati (immagini) 8 | Lento, richiede pre-processing immagini, installazione binari esterni | Gestione eccezioni per PDF non testuali |

La raccomandazione per questo progetto è l'utilizzo di **PyMuPDF**. La sua capacità di estrarre testo in formati strutturati (come HTML o blocchi con coordinate) permette di filtrare header e footer ricorrenti che inquinerebbero il riassunto.9 Inoltre, consente di identificare la formattazione (grassetto, dimensione font) per inferire la gerarchia dei titoli, fondamentale per una segmentazione semantica del documento.

Per i PDF scansionati, l'integrazione di una pipeline OCR è mandatoria. Librerie come pdf2image possono convertire le pagine in immagini, che vengono poi processate da pytesseract o servizi cloud come Amazon Textract.8 Tuttavia, dato il focus accademico, l'uso di Tesseract locale è l'approccio più sostenibile e formativo.

#### **2.1.2 Scraping Intelligente di Documentazione Web (ReadTheDocs, Docusaurus)**

Le piattaforme di documentazione moderna come ReadTheDocs (basata su Sphinx) e Docusaurus generano siti statici con strutture prevedibili ma complesse. Uno scraper generico rischia di catturare menu di navigazione, barre laterali e footer, diluendo il contenuto informativo.

Per ReadTheDocs e siti simili, l'approccio ottimale prevede:

1. **Identificazione della Sitemap:** La maggior parte di questi siti espone una sitemap.xml che elenca tutte le pagine. Utilizzare un loader basato su sitemap (come SitemapLoader di LangChain) garantisce la copertura completa.11  
2. **Main Content Extraction:** L'uso di librerie specializzate come **Trafilatura** è superiore a BeautifulSoup per questo task. Trafilatura utilizza euristiche avanzate per identificare il corpo principale del testo, scartando il boilerplate navigazionale, e preserva la struttura semantica (paragrafi, titoli) meglio di alternative come Newspaper3k.12  
3. **Parsing Gerarchico dell'HTML:** Se si deve intervenire manualmente, il parsing dell'HTML generato da Sphinx o Docusaurus deve mirare a specifici div o tag article (es. div.document in Sphinx o article in Docusaurus). Librerie come parsel o lxml offrono selettori CSS e XPath potenti per questo scopo.15

Un aspetto cruciale è la gestione dei **blocchi di codice**. Nella documentazione API, il codice è spesso preponderante. Per un riassunto concettuale, il codice è spesso superfluo e consuma preziosi token. Una strategia efficace è rilevare i tag \<pre\> o \<code\> e sostituirli con un token segnaposto (es. \`\`), oppure estrarne solo i commenti se il linguaggio lo permette.17

#### **2.1.3 Parsing Semantico di Specifiche OpenAPI (Swagger)**

Le specifiche OpenAPI (precedentemente Swagger) rappresentano lo standard per la descrizione di API RESTful. Sono file strutturati (JSON o YAML) che descrivono endpoint, metodi, parametri e schemi di risposta.19 Trattare questi file come testo libero è un errore metodologico grave.

L'approccio corretto richiede un parser semantico che comprenda la struttura OpenAPI. Librerie come swagger-parser, prance o openapi3-parser permettono di caricare la specifica come oggetto Python e navigarla programmaticamente.21  
Il processo di estrazione per la sintesi dovrebbe focalizzarsi sui campi descrittivi ("discorsivi") della specifica:

* info.description: Descrizione generale dell'API.  
* paths.{path}.{method}.summary e description: Spiegazione di cosa fa ogni endpoint.  
* components.schemas.{schema}.description: Dettagli sui modelli di dati.

È possibile generare un testo sintetico ("pseudo-documento") iterando su questi campi. Ad esempio: *"L'endpoint GET /users permette di recuperare la lista utenti. Accetta parametri opzionali per il filtraggio..."*. Questo testo generato, molto più denso e rilevante del JSON grezzo, sarà l'input per il modello di summarization.22

### **2.2 Pre-processing Avanzato per il Dominio Tecnico**

Una volta acquisito il testo grezzo, una fase di pre-processing specifica è necessaria per prepararlo alla modellazione.

#### **2.2.1 Pulizia e Normalizzazione**

Oltre alle operazioni standard (rimozione spazi multipli, normalizzazione Unicode), i testi tecnici richiedono attenzioni particolari:

* **Preservazione della Case:** In ambito tecnico, la differenza tra String e string, o GET e get, può essere semanticamente rilevante. A differenza del sentiment analysis, qui il *lowercasing* indiscriminato va evitato.  
* **Gestione di URL e Path:** Percorsi di file e URL dovrebbero essere normalizzati o sostituiti con token speciali se non essenziali per il riassunto.

#### **2.2.2 Strategie di Chunking (Segmentazione)**

I modelli Transformer hanno limiti rigidi sulla lunghezza della sequenza di input (es. 512 token per BERT, 1024 per T5 base). La documentazione API supera quasi sempre questi limiti.  
Il Chunking non deve essere arbitrario. Tagliare una frase a metà o separare un titolo dal suo paragrafo degrada la qualità del riassunto.  
Si raccomanda l'uso di RecursiveCharacterTextSplitter (disponibile in LangChain), che tenta di dividere il testo usando separatori gerarchici (doppio a capo per paragrafi, a capo singolo per righe, spazi per parole).24  
Per documenti Markdown o strutturati, un Structure-aware splitting (es. MarkdownTextSplitter) è ancora meglio: divide il testo rispettando le sezioni logiche (Header 1, Header 2), garantendo che ogni chunk sia un'unità informativa coerente.25

## **3\. Modulo di Sintesi Documentale (Text Summarization)**

La summarization è il processo di distillazione delle informazioni chiave da un testo sorgente.

### **3.1 Approcci Teorici: Estrattivo vs Astrattivo**

La letteratura distingue nettamente tra due paradigmi:

1. **Summarization Estrattiva:** Seleziona un sottoinsieme di frasi esistenti nel testo originale che meglio ne rappresentano il contenuto. È un problema di ranking. Algoritmi come **TextRank** (basato su grafi, simile a PageRank) o approcci basati su clustering di embedding (es. BERT-Extractive-Summarizer) sono molto efficaci per ottenere riassunti fattualmente corretti e veloci da generare.26 Tuttavia, il risultato può mancare di coerenza discorsiva.  
2. **Summarization Astrattiva:** Genera nuovo testo, riformulando e condensando i concetti. Richiede una comprensione profonda (NLU) e capacità di generazione (NLG). Modelli Seq2Seq come T5, BART e PEGASUS dominano questo campo.2 Offrono riassunti più fluidi e naturali, ma introducono il rischio di "allucinazioni" fattuali.

Per la documentazione tecnica, dove la precisione è fondamentale ma la leggibilità è desiderata, un approccio **Astrattivo** basato su modelli potenti è generalmente preferito oggi, a patto di utilizzare modelli ben addestrati. Un'alternativa ibrida consiste nell'usare un metodo estrattivo per selezionare le parti rilevanti e poi un modello astrattivo per parafrasarle.31

### **3.2 Modelli Transformer per la Sintesi: Focus su T5 e IT5**

Per questo progetto, il modello di riferimento suggerito è T5 (Text-to-Text Transfer Transformer). La sua architettura unificata permette di trattare la sintesi semplicemente prependedo il prefisso "summarize: " all'input.  
Poiché il requisito implica il supporto per l'italiano, non possiamo affidarci ai checkpoint standard di T5 (addestrati su C4 inglese). Dobbiamo utilizzare IT5 (Italian T5), una variante pre-addestrata su un massiccio corpus italiano pulito.32  
Specificamente, il modello gsarti/it5-small-wiki-summarization o efederici/it5-base-summarization su Hugging Face offre un eccellente compromesso tra prestazioni e risorse computazionali, essendo già stato fine-tunato su task di sintesi.32  
Per chi desidera prestazioni multilingue o state-of-the-art su testi lunghi, **BART** (e la sua variante multilingue mBART) è una valida alternativa, specialmente il modello facebook/bart-large-cnn per l'inglese, ma per l'italiano specifico IT5 resta superiore in termini di coerenza linguistica.34

### **3.3 Gestione del Contesto Lungo: Map-Reduce**

Anche con un buon chunking, ci troviamo con molteplici frammenti di testo. Come ottenere un riassunto unico? La tecnica **Map-Reduce** è lo standard industriale per questo problema.35

1. **Map Step:** Ogni chunk del documento viene passato indipendentemente al modello di sintesi (es. IT5) per generare un "micro-riassunto".  
2. Reduce Step: I micro-riassunti vengono concatenati. Se la lunghezza totale è gestibile, vengono passati nuovamente al modello per generare il riassunto finale. Se è ancora troppo lunga, si applica un secondo livello di sintesi ricorsiva.  
   Librerie come LangChain offrono catene pre-costruite (load\_summarize\_chain con chain\_type="map\_reduce") che astraggono completamente questa complessità logistica.35

### **3.4 Metriche di Valutazione della Sintesi**

Come valutiamo oggettivamente la qualità del riassunto?

* **ROUGE (Recall-Oriented Understudy for Gisting Evaluation):** È la metrica standard. Confronta gli n-grammi del riassunto generato con quelli di un riassunto di riferimento (gold standard).  
  * *ROUGE-1*: Overlap di unigrammi (parole singole). Misura l'informatività.  
  * *ROUGE-2*: Overlap di bigrammi. Misura la fluidità.  
  * *ROUGE-L*: Longest Common Subsequence. Misura la struttura della frase.36  
* BERTScore: Calcola la similarità semantica tra le frasi usando embedding contestuali, superando il limite del matching esatto di parole di ROUGE. È più correlato al giudizio umano.38  
  In assenza di riassunti di riferimento per la documentazione API generica (scenario unsupervised evaluation), la valutazione quantitativa è difficile. Si può ricorrere a metriche di consistency o coverage basate su entailment (verificare se il riassunto è logicamente implicato dal testo) o valutazioni umane qualitative.39

## **4\. Modulo di Analisi del Sentiment su Recensioni Utente**

L'analisi del sentiment è un compito di classificazione supervisionata. L'obiettivo è assegnare un'etichetta (Positivo, Negativo, Neutro) a un testo.

### **4.1 Specificità del Pre-processing per Social Media e Recensioni**

Le recensioni utente differiscono drasticamente dalla documentazione tecnica. Sono informali, rumorose e spesso sgrammaticate.

* **Gestione Emoji:** Le emoji sono vettori densi di sentiment. Rimuoverle è un errore. Librerie come emoji in Python possono "demojizzare" il testo (es. convertire "😊" in ":smile:"), rendendolo interpretabile dal modello.40  
* **Normalizzazione User/URL:** In recensioni o tweet, menzioni (@utente) e link non portano sentiment. Vanno sostituiti con token generici (@USER, HTTPURL) per ridurre la sparsità del vocabolario senza perdere la struttura della frase.41  
* **Lemmatizzazione:** Con i moderni modelli Transformer (BERT), la lemmatizzazione (ridurre le parole alla radice) è spesso sconsigliata perché può rimuovere suffissi che portano significato grammaticale o di tono. BERT utilizza tokenizzatori a sottoparole (WordPiece) che gestiscono efficacemente le varianti morfologiche.

### **4.2 L'Egemonia dei Transformer: BERT e le Varianti Italiane**

L'approccio *bag-of-words* o le vecchie RNN (LSTM) sono oggi superati dai Transformer per compiti di classificazione. Il meccanismo di *self-attention* permette a BERT di comprendere il contesto bidirezionale di una parola (fondamentale per capire negazioni come "non mi è piaciuto affatto").42

Per l'italiano, il "Fine-Tuning" di modelli pre-addestrati è la via maestra:

* **UmBERTo:** Basato su RoBERTa, addestrato su un corpus massiccio (OSCAR) in italiano. È estremamente robusto per l'italiano generale.44  
* **AlBERTo:** Un modello BERT addestrato specificamente su tweet italiani. È ideale se le recensioni sono molto brevi e colloquiali (stile social media).45  
* **FEEL-IT:** Un modello basato su UmBERTo e fine-tunato specificamente per il riconoscimento di emozioni e sentiment su dataset italiani (Wita). È lo stato dell'arte open-source per questo task.46 Utilizzare FEEL-IT (MilaNLProc/feel-it-italian-sentiment su Hugging Face) permette di ottenere prestazioni elevate *out-of-the-box* senza dover addestrare un modello da zero.

### **4.3 Strategie per Classi Sbilanciate**

I dati reali di recensioni sono quasi sempre sbilanciati (es. 80% recensioni positive, 10% negative, 10% neutre). Un modello addestrato su questi dati tenderà a ignorare le classi minoritarie.  
Soluzioni tecniche da implementare:

1. **Resampling:**  
   * *Undersampling* della classe maggioritaria (rischio di perdita dati).  
   * *Oversampling* della minoritaria (duplicazione esempi).  
2. **Generazione Sintetica:** Tecniche come **SMOTE** (Synthetic Minority Over-sampling Technique) creano nuovi esempi sintetici interpolando nello spazio delle feature, anche se la sua applicazione diretta su embedding testuali ad alta dimensionalità è complessa.3  
3. **Class Weights (Pesi delle Classi):** La soluzione più elegante. Si calcolano pesi inversamente proporzionali alla frequenza delle classi e si passano alla funzione di perdita (*Loss Function*) del modello (es. CrossEntropyLoss in PyTorch). Questo penalizza maggiormente gli errori sulle classi rare, forzando il modello ad apprenderle.4

### **4.4 Metriche di Valutazione per la Classificazione**

Mai affidarsi alla sola **Accuracy** su dati sbilanciati (un modello che predice sempre "Positivo" su un dataset al 90% positivo avrebbe un'accuratezza del 90% ma sarebbe inutile).

* **F1-Score (Macro/Weighted):** La media armonica di Precision e Recall. È la metrica regina per classi sbilanciate.49  
* **Matrice di Confusione:** Strumento diagnostico fondamentale per vedere quali classi vengono confuse (es. il modello confonde spesso "Neutro" con "Negativo"?).  
* **AUC-ROC:** Utile per valutare la capacità di discriminazione del modello a diverse soglie di probabilità.

## **5\. Piano di Implementazione Pratica e Architettura Software**

La realizzazione del progetto deve seguire standard di ingegneria del software moderni per garantire riproducibilità e manutenibilità.

### **5.1 Stack Tecnologico e Dipendenze**

* **Linguaggio:** Python 3.9+ (standard *de facto* per AI/NLP).  
* **Core NLP/DL:** transformers (Hugging Face), torch (PyTorch).  
* **Data Manipulation:** pandas, numpy.  
* **PDF Processing:** pymupdf (importato come fitz).  
* **Web/API Extraction:** trafilatura (web), swagger-parser (OpenAPI).  
* **UI Framework:** streamlit (per prototipazione rapida di data apps).  
* **Environment:** conda o venv per isolamento, pip per gestione pacchetti.

### **5.2 Struttura del Progetto (Project Layout)**

Una struttura ordinata è cruciale.50 Si consiglia il seguente layout:

nlp\_project/  
├── data/ \# Dati grezzi e processati (ignorati da git)  
│ ├── raw/  
│ ├── processed/  
│ └── external/  
├── models/ \# Checkpoint dei modelli salvati  
├── notebooks/ \# Jupyter notebooks per esplorazione e analisi (EDA)  
├── src/ \# Codice sorgente principale  
│ ├── init.py  
│ ├── data\_ingestion.py \# Funzioni per caricare PDF, URL, JSON  
│ ├── preprocessing.py \# Pulizia testo, chunking, tokenizzazione  
│ ├── summarization.py \# Logica inferenza T5/IT5  
│ ├── sentiment.py \# Logica inferenza BERT/FEEL-IT  
│ └── utils.py \# Funzioni di utilità, logging, config  
├── app.py \# Entry point applicazione Streamlit  
├── config.yaml \# Configurazioni (path, parametri modelli)  
├── requirements.txt \# Dipendenze pip  
└── README.md \# Documentazione progetto

### **5.3 Dettaglio Implementativo dei Moduli**

#### **5.3.1 Modulo Ingestion (src/data\_ingestion.py)**

Deve implementare una logica *Factory* o *Dispatcher* che seleziona la strategia di estrazione in base al tipo di input.

* **Funzione extract\_from\_pdf(path):** Utilizza pymupdf per iterare sulle pagine. Implementare euristiche spaziali per escludere header/footer (es. escludere testo nel top/bottom 10% della pagina).52  
* **Funzione extract\_from\_url(url):** Utilizza trafilatura.fetch\_url() e extract() per ottenere il contenuto principale. Gestire errori di connessione e timeout.  
* **Funzione parse\_openapi(json\_path):** Utilizza swagger\_parser per caricare il JSON. Iterare su paths e concatenare summary, description, e descrizioni dei parametri in un unico testo coerente, separando gli endpoint con newline.

#### **5.3.2 Modulo Summarization (src/summarization.py)**

Utilizzare la classe pipeline di Hugging Face per astrazione.

Python

from transformers import pipeline

class Summarizer:  
    def \_\_init\_\_(self, model\_name="gsarti/it5-small-wiki-summarization"):  
        self.summarizer \= pipeline("summarization", model=model\_name, tokenizer=model\_name)

    def summarize(self, text, max\_chunk\_len=1024):  
        \# Implementare qui la logica di chunking (es. con LangChain)  
        \# e la logica Map-Reduce per generare il riassunto finale  
        pass

È cruciale gestire la lunghezza massima (truncation=True) e ottimizzare parametri di generazione come num\_beams (per beam search) e length\_penalty per controllare la verbosità del riassunto.53

#### **5.3.3 Modulo Sentiment (src/sentiment.py)**

Caricamento ottimizzato del modello FEEL-IT.

Python

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class SentimentAnalyzer:  
    def \_\_init\_\_(self, model\_name="MilaNLProc/feel-it-italian-sentiment"):  
        self.classifier \= pipeline("text-classification", model=model\_name, return\_all\_scores=True)

    def analyze(self, texts):  
        \# Inferenza batch per efficienza su liste di recensioni  
        return self.classifier(texts)

La funzione deve restituire non solo la classe (es. "negative") ma anche lo score di confidenza, utile per visualizzare l'intensità del sentimento nella UI.

### **5.4 Interfaccia Utente con Streamlit (app.py)**

Streamlit permette di trasformare gli script Python in web app interattive con sforzo minimo.54  
L'app dovrebbe essere strutturata come una Multipage App o usare una Sidebar per la navigazione.56

* **Home/Dashboard:** Panoramica del progetto.  
* **Pagina 1: Sintesi Documentazione:**  
  * Widget st.text\_input per URL o st.file\_uploader per PDF/JSON.  
  * Pulsante "Processa".  
  * Visualizzazione del testo estratto (in un st.expander per non ingombrare) per verifica.  
  * Visualizzazione del Riassunto finale in evidenza.  
* **Pagina 2: Analisi Sentiment:**  
  * st.file\_uploader per caricare un CSV di recensioni.  
  * Analisi automatica al caricamento.  
  * **Visualizzazioni:**  
    * Grafico a torta (Pie Chart) della distribuzione Positivo/Negativo/Neutro (usando plotly o altair integrati in Streamlit).  
    * WordCloud delle parole più frequenti per ogni classe di sentiment.  
    * Dataframe esplorabile con le recensioni e il loro score.

## **6\. Valutazione e Iterazione del Progetto**

Il ciclo di vita del progetto non termina con l'implementazione.

* **Validazione della Sintesi:** Data la mancanza di "ground truth" per la documentazione API generica, si consiglia una **valutazione qualitativa umana** su un piccolo campione (es. 10 documenti), valutando Coerenza, Rilevanza e Assenza di Allucinazioni su scala 1-5.57 In ambito accademico, l'uso di metriche "reference-free" basate su LLM (come G-Eval con GPT-4) sta emergendo come standard.38  
* **Validazione del Sentiment:** Calcolare accuratezza e F1-score sul dataset di recensioni disponibile (usando un hold-out set o cross-validation). Analizzare la matrice di confusione per capire dove il modello fallisce (spesso su recensioni miste o sarcastiche) e considerare un ulteriore fine-tuning se le performance sono insufficienti (\< 80% F1).

## **7\. Conclusioni**

Questo piano d'implementazione fornisce una roadmap tecnica dettagliata per la costruzione di un sistema NLP avanzato. Sfruttando modelli Transformer specifici per l'italiano (IT5, FEEL-IT) e librerie di elaborazione dati robuste (PyMuPDF, Trafilatura, Swagger-Parser), è possibile superare le sfide poste dalla natura eterogenea dei dati documentali e delle recensioni. L'architettura modulare proposta garantisce scalabilità e manutenibilità, mentre l'uso di Streamlit assicura un'interfaccia utente professionale e accessibile. L'adozione di metriche di valutazione rigorose e strategie di gestione dei dati sbilanciati e dei testi lunghi eleva il progetto da semplice esercizio di codifica a una soluzione ingegneristica completa e matura.

#### **Bibliografia**

1. Parte 11 \- La pipeline NLP.pdf  
2. Tech Deep Dive: Extractive vs. abstractive summaries and how machines write them \- Iris.ai, accesso eseguito il giorno dicembre 12, 2025, [https://iris.ai/blog/tech-deep-dive-extractive-vs-abstractive-summaries-and-how-machines-write-them](https://iris.ai/blog/tech-deep-dive-extractive-vs-abstractive-summaries-and-how-machines-write-them)  
3. How to deal with data imbalance problem in sentiment analysis? \- Tencent Cloud, accesso eseguito il giorno dicembre 12, 2025, [https://www.tencentcloud.com/techpedia/106765](https://www.tencentcloud.com/techpedia/106765)  
4. Fine-Tuning BERT for Sentiment Analysis: A Practical Guide | by Hey Amit | Medium, accesso eseguito il giorno dicembre 12, 2025, [https://medium.com/@heyamit10/fine-tuning-bert-for-sentiment-analysis-a-practical-guide-f3d9c9cac236](https://medium.com/@heyamit10/fine-tuning-bert-for-sentiment-analysis-a-practical-guide-f3d9c9cac236)  
5. Extract text from PDF File using Python \- GeeksforGeeks, accesso eseguito il giorno dicembre 12, 2025, [https://www.geeksforgeeks.org/python/extract-text-from-pdf-file-using-python/](https://www.geeksforgeeks.org/python/extract-text-from-pdf-file-using-python/)  
6. PyMuPDF4LLM \- PyMuPDF documentation \- Read the Docs, accesso eseguito il giorno dicembre 12, 2025, [https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)  
7. Extracting text from PDF files with Python: A comprehensive guide | Towards Data Science, accesso eseguito il giorno dicembre 12, 2025, [https://towardsdatascience.com/extracting-text-from-pdf-files-with-python-a-comprehensive-guide-9fc4003d517/](https://towardsdatascience.com/extracting-text-from-pdf-files-with-python-a-comprehensive-guide-9fc4003d517/)  
8. Python OCR libraries for converting PDFs into editable text \- Ploomber, accesso eseguito il giorno dicembre 12, 2025, [https://ploomber.io/blog/pdf-ocr/](https://ploomber.io/blog/pdf-ocr/)  
9. Text \- PyMuPDF documentation, accesso eseguito il giorno dicembre 12, 2025, [https://pymupdf.readthedocs.io/en/latest/recipes-text.html](https://pymupdf.readthedocs.io/en/latest/recipes-text.html)  
10. Appendix 1: Details on Text Extraction \- PyMuPDF documentation, accesso eseguito il giorno dicembre 12, 2025, [https://pymupdf.readthedocs.io/en/latest/app1.html](https://pymupdf.readthedocs.io/en/latest/app1.html)  
11. Docusaurus \- Docs by LangChain, accesso eseguito il giorno dicembre 12, 2025, [https://docs.langchain.com/oss/python/integrations/document\_loaders/docusaurus](https://docs.langchain.com/oss/python/integrations/document_loaders/docusaurus)  
12. A Python package & command-line tool to gather text on the Web — Trafilatura 2.0.0 documentation, accesso eseguito il giorno dicembre 12, 2025, [https://trafilatura.readthedocs.io/](https://trafilatura.readthedocs.io/)  
13. Scraping Web Page Content with Python: Trafilatura, Readability, Newspaper3k & Playwright | JustToThePoint, accesso eseguito il giorno dicembre 12, 2025, [https://www.justtothepoint.com/code/scrape/](https://www.justtothepoint.com/code/scrape/)  
14. An Evaluation of Main Content Extraction Libraries in Java and Python \- OSTI, accesso eseguito il giorno dicembre 12, 2025, [https://www.osti.gov/servlets/purl/2429881](https://www.osti.gov/servlets/purl/2429881)  
15. Parsel: How to Extract Text From HTML in Python \- WebScrapingAPI, accesso eseguito il giorno dicembre 12, 2025, [https://www.webscrapingapi.com/how-to-extract-text-from-html-in-python](https://www.webscrapingapi.com/how-to-extract-text-from-html-in-python)  
16. html-text \- PyPI, accesso eseguito il giorno dicembre 12, 2025, [https://pypi.org/project/html-text/](https://pypi.org/project/html-text/)  
17. Enhancing API Documentation through BERTopic Modeling and Summarization \- arXiv, accesso eseguito il giorno dicembre 12, 2025, [https://arxiv.org/pdf/2308.09070](https://arxiv.org/pdf/2308.09070)  
18. Revolutionizing API Documentation through Summarization \- arXiv, accesso eseguito il giorno dicembre 12, 2025, [https://arxiv.org/html/2401.11361v1](https://arxiv.org/html/2401.11361v1)  
19. OpenAPI Specification \- Version 3.1.0 \- Swagger, accesso eseguito il giorno dicembre 12, 2025, [https://swagger.io/specification/](https://swagger.io/specification/)  
20. OpenAPI docs \- FastAPI, accesso eseguito il giorno dicembre 12, 2025, [https://fastapi.tiangolo.com/reference/openapi/docs/](https://fastapi.tiangolo.com/reference/openapi/docs/)  
21. swagger-parser \- PyPI, accesso eseguito il giorno dicembre 12, 2025, [https://pypi.org/project/swagger-parser/](https://pypi.org/project/swagger-parser/)  
22. Natural Language Sentence Generation from API Specifications \- ResearchGate, accesso eseguito il giorno dicembre 12, 2025, [https://www.researchgate.net/publication/361300735\_Natural\_Language\_Sentence\_Generation\_from\_API\_Specifications](https://www.researchgate.net/publication/361300735_Natural_Language_Sentence_Generation_from_API_Specifications)  
23. ShelbyJenkins/LLM-OpenAPI-minifier: Making openapi spec swagger documents friendly for GPT and other LLMs. \- GitHub, accesso eseguito il giorno dicembre 12, 2025, [https://github.com/ShelbyJenkins/LLM-OpenAPI-minifier](https://github.com/ShelbyJenkins/LLM-OpenAPI-minifier)  
24. Text Splitter in LangChain \- GeeksforGeeks, accesso eseguito il giorno dicembre 12, 2025, [https://www.geeksforgeeks.org/artificial-intelligence/text-splitter-in-langchain/](https://www.geeksforgeeks.org/artificial-intelligence/text-splitter-in-langchain/)  
25. Text splitters \- Docs by LangChain, accesso eseguito il giorno dicembre 12, 2025, [https://docs.langchain.com/oss/python/integrations/splitters](https://docs.langchain.com/oss/python/integrations/splitters)  
26. Text Summarization in NLP \- GeeksforGeeks, accesso eseguito il giorno dicembre 12, 2025, [https://www.geeksforgeeks.org/nlp/text-summarization-in-nlp/](https://www.geeksforgeeks.org/nlp/text-summarization-in-nlp/)  
27. Text Summarization using the TextRank Algorithm (with Python) \- Analytics Vidhya, accesso eseguito il giorno dicembre 12, 2025, [https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/](https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/)  
28. Biased TextRank: Unsupervised Graph-Based Content Extraction \- ACL Anthology, accesso eseguito il giorno dicembre 12, 2025, [https://aclanthology.org/2020.coling-main.144.pdf](https://aclanthology.org/2020.coling-main.144.pdf)  
29. AI Summarization: Extractive and Abstractive Techniques \- DZone, accesso eseguito il giorno dicembre 12, 2025, [https://dzone.com/articles/ai-summarization-extractive-abstractive-techniques](https://dzone.com/articles/ai-summarization-extractive-abstractive-techniques)  
30. Two minutes NLP — Four different approaches to Text Summarization | by Fabio Chiusano | Generative AI | Medium, accesso eseguito il giorno dicembre 12, 2025, [https://medium.com/nlplanet/two-minutes-nlp-four-different-approaches-to-text-summarization-5a0ce9c06c74](https://medium.com/nlplanet/two-minutes-nlp-four-different-approaches-to-text-summarization-5a0ce9c06c74)  
31. Text Summarization in NLP: Key Concepts, Techniques and Implementation \- upGrad, accesso eseguito il giorno dicembre 12, 2025, [https://www.upgrad.com/blog/text-summarization-in-nlp/](https://www.upgrad.com/blog/text-summarization-in-nlp/)  
32. gsarti/it5-small-wiki-summarization \- Hugging Face, accesso eseguito il giorno dicembre 12, 2025, [https://huggingface.co/gsarti/it5-small-wiki-summarization](https://huggingface.co/gsarti/it5-small-wiki-summarization)  
33. efederici/it5-base-summarization \- Hugging Face, accesso eseguito il giorno dicembre 12, 2025, [https://huggingface.co/efederici/it5-base-summarization](https://huggingface.co/efederici/it5-base-summarization)  
34. Two New Datasets for Italian-Language Abstractive Text Summarization \- MDPI, accesso eseguito il giorno dicembre 12, 2025, [https://www.mdpi.com/2078-2489/13/5/228](https://www.mdpi.com/2078-2489/13/5/228)  
35. How to summarize large documents : r/LangChain \- Reddit, accesso eseguito il giorno dicembre 12, 2025, [https://www.reddit.com/r/LangChain/comments/1hxeqev/how\_to\_summarize\_large\_documents/](https://www.reddit.com/r/LangChain/comments/1hxeqev/how_to_summarize_large_documents/)  
36. LLM Evaluation For Text Summarization \- Neptune.ai, accesso eseguito il giorno dicembre 12, 2025, [https://neptune.ai/blog/llm-evaluation-text-summarization](https://neptune.ai/blog/llm-evaluation-text-summarization)  
37. Evaluation Metrics for Summarization \- DEV Community, accesso eseguito il giorno dicembre 12, 2025, [https://dev.to/espoir/evaluation-metrics-for-summarization-3amo](https://dev.to/espoir/evaluation-metrics-for-summarization-3amo)  
38. How to evaluate a summarization task | OpenAI Cookbook, accesso eseguito il giorno dicembre 12, 2025, [https://cookbook.openai.com/examples/evaluation/how\_to\_eval\_abstractive\_summarization](https://cookbook.openai.com/examples/evaluation/how_to_eval_abstractive_summarization)  
39. A Step-By-Step Guide to Evaluating an LLM Text Summarization Task \- Confident AI, accesso eseguito il giorno dicembre 12, 2025, [https://www.confident-ai.com/blog/a-step-by-step-guide-to-evaluating-an-llm-text-summarization-task](https://www.confident-ai.com/blog/a-step-by-step-guide-to-evaluating-an-llm-text-summarization-task)  
40. Sentiment analysis in Python (Complete guide for 2025\) \- Apify Blog, accesso eseguito il giorno dicembre 12, 2025, [https://blog.apify.com/sentiment-analysis-python/](https://blog.apify.com/sentiment-analysis-python/)  
41. neuraly/bert-base-italian-cased-sentiment \- Hugging Face, accesso eseguito il giorno dicembre 12, 2025, [https://huggingface.co/neuraly/bert-base-italian-cased-sentiment](https://huggingface.co/neuraly/bert-base-italian-cased-sentiment)  
42. Fine-Tuning BERT for Sentiment Analysis: Boost Accuracy \- HERE AND NOW AI, accesso eseguito il giorno dicembre 12, 2025, [https://hereandnowai.com/fine-tuning-bert-for-sentiment-analysis/](https://hereandnowai.com/fine-tuning-bert-for-sentiment-analysis/)  
43. How to Fine-Tune BERT for Sentiment Analysis with Hugging Face Transformers, accesso eseguito il giorno dicembre 12, 2025, [https://www.kdnuggets.com/how-to-fine-tune-bert-sentiment-analysis-hugging-face-transformers](https://www.kdnuggets.com/how-to-fine-tune-bert-sentiment-analysis-hugging-face-transformers)  
44. Sentiment Analysis and Emotion Recognition in Italian (using BERT), accesso eseguito il giorno dicembre 12, 2025, [https://towardsdatascience.com/sentiment-analysis-and-emotion-recognition-in-italian-using-bert-92f5c8fe8a2/](https://towardsdatascience.com/sentiment-analysis-and-emotion-recognition-in-italian-using-bert-92f5c8fe8a2/)  
45. Comparative Approaches to Sentiment Analysis Using Datasets in Major European and Arabic Languages \- arXiv, accesso eseguito il giorno dicembre 12, 2025, [https://arxiv.org/pdf/2501.12540](https://arxiv.org/pdf/2501.12540)  
46. Italian NLP Resources \- a gsarti Collection \- Hugging Face, accesso eseguito il giorno dicembre 12, 2025, [https://huggingface.co/collections/gsarti/italian-nlp-resources](https://huggingface.co/collections/gsarti/italian-nlp-resources)  
47. MilaNLProc/feel-it: Sentiment analysis and emotion classification for Italian using BERT (fine-tuning). Published at the WASSA workshop (EACL2021). \- GitHub, accesso eseguito il giorno dicembre 12, 2025, [https://github.com/MilaNLProc/feel-it](https://github.com/MilaNLProc/feel-it)  
48. Imbalanced Review Sentiment Classification \- Kaggle, accesso eseguito il giorno dicembre 12, 2025, [https://www.kaggle.com/code/chandlerunderwood/imbalanced-review-sentiment-classification](https://www.kaggle.com/code/chandlerunderwood/imbalanced-review-sentiment-classification)  
49. urllib.parse — Parse URLs into components — Python 3.14.2 documentation, accesso eseguito il giorno dicembre 12, 2025, [https://docs.python.org/3/library/urllib.parse.html](https://docs.python.org/3/library/urllib.parse.html)  
50. A Light and Modular PyTorch NLP Project Template \- GitHub, accesso eseguito il giorno dicembre 12, 2025, [https://github.com/ahmetgunduz/pytorch-nlp-project-template](https://github.com/ahmetgunduz/pytorch-nlp-project-template)  
51. Best Practices in Structuring Python Projects \- Dagster, accesso eseguito il giorno dicembre 12, 2025, [https://dagster.io/blog/python-project-best-practices](https://dagster.io/blog/python-project-best-practices)  
52. Extract Text from a PDF — pypdf 6.4.1 documentation \- Read the Docs, accesso eseguito il giorno dicembre 12, 2025, [https://pypdf.readthedocs.io/en/stable/user/extract-text.html](https://pypdf.readthedocs.io/en/stable/user/extract-text.html)  
53. Text Summarization using T5: Fine-Tuning and Building Gradio App \- LearnOpenCV, accesso eseguito il giorno dicembre 12, 2025, [https://learnopencv.com/text-summarization-using-t5/](https://learnopencv.com/text-summarization-using-t5/)  
54. Gradio vs. Streamlit: Which App Builder Won't Break Your Brain? \- Sider.AI, accesso eseguito il giorno dicembre 12, 2025, [https://sider.ai/blog/ai-tools/gradio-vs\_streamlit-which-app-builder-won-t-break-your-brain](https://sider.ai/blog/ai-tools/gradio-vs_streamlit-which-app-builder-won-t-break-your-brain)  
55. Streamlit vs Gradio (and More): Building ML Web Apps | by Saiii \- Medium, accesso eseguito il giorno dicembre 12, 2025, [https://medium.com/@sailakkshmiallada/streamlit-vs-gradio-and-more-building-ml-web-apps-6753f5147276](https://medium.com/@sailakkshmiallada/streamlit-vs-gradio-and-more-building-ml-web-apps-6753f5147276)  
56. Create a multipage app \- Streamlit Docs, accesso eseguito il giorno dicembre 12, 2025, [https://docs.streamlit.io/get-started/tutorials/create-a-multipage-app](https://docs.streamlit.io/get-started/tutorials/create-a-multipage-app)  
57. Evaluating LLMs for Text Summarization: An Introduction \- Software Engineering Institute, accesso eseguito il giorno dicembre 12, 2025, [https://www.sei.cmu.edu/blog/evaluating-llms-for-text-summarization-introduction/](https://www.sei.cmu.edu/blog/evaluating-llms-for-text-summarization-introduction/)