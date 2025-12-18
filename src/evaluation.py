import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import evaluate
import pandas as pd
from summarization import SummarizerModule
from tabulate import tabulate



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_manual_test_data():
    """
    Returns a small manual dataset for evaluation to avoid dependency hell with 
    deprecated dataset scripts in newer 'datasets' library versions.
    """
    return [
        {
            "text": "L'intelligenza artificiale generativa sta trasformando profondamente il mondo del lavoro in ogni suo settore. Strumenti avanzati come ChatGPT, Claude e Midjourney permettono di creare contenuti testuali, codice informatico e immagini realistiche in pochi secondi, aumentando la produttività. Tuttavia, sorgono forti preoccupazioni etiche riguardanti il copyright delle opere usate per l'addestramento e la possibile sostituzione di posti di lavoro umani, specialmente nei settori creativi. Gli esperti suggeriscono che l'IA non sostituirà l'uomo, ma chi usa l'IA in modo efficace sostituirà chi non la usa. È fondamentale una regolamentazione internazionale chiara per garantire un utilizzo responsabile, trasparente ed equo di queste tecnologie rivoluzionarie.",
            "summary": "L'IA generativa rivoluziona il lavoro ma solleva dubbi etici e occupazionali. Gli esperti ritengono essenziale l'integrazione umana e una regolamentazione adeguata."
        },
        {
            "text": "La dieta mediterranea è universalmente considerata dai medici una delle migliori al mondo per mantenere una buona salute cardiovascolare e generale. Basata su un elevato consumo quotidiano di frutta fresca, verdura di stagione, legumi, cereali integrali e olio extravergine d'oliva, limita invece drasticamente le carni rosse processate e i grassi saturi dannosi. Numerosi studi scientifici decennali hanno dimostrato che questo regime riduce significativamente il rischio di infarto, ictus, diabete di tipo 2 e obesità. Inoltre, rappresenta un modello alimentare altamente sostenibile per l'ambiente grazie al basso impatto ecologico delle produzioni vegetali rispetto agli allevamenti intensivi.",
            "summary": "La dieta mediterranea, ricca di vegetali e povera di grassi saturi, protegge il cuore e previene malattie, rappresentando anche una scelta sostenibile."
        },
        {
            "text": "Il cambiamento climatico sta causando eventi meteorologici sempre più estremi e frequenti in tutto il globo terrestre. Ondate di calore record, alluvioni improvvise devastanti e siccità prolungate stanno mettendo a dura prova l'agricoltura globale e la tenuta delle infrastrutture urbane. La conferenza internazionale COP28 ha stabilito nuovi obiettivi ambiziosi per la riduzione delle emissioni di CO2, ma molti scienziati ritengono che le azioni politiche attuali siano ancora insufficienti per evitare il peggio. È urgente una transizione energetica rapida e radicale verso fonti rinnovabili come sole e vento per limitare il riscaldamento globale entro la soglia critica di 1.5 gradi centigradi.",
            "summary": "Il cambiamento climatico provoca eventi estremi che danneggiano agricoltura e città. Nonostante gli accordi COP28, serve una transizione urgente alle rinnovabili per limitare il riscaldamento."
        },
        {
            "text": "Kubernetes è diventato lo standard de facto per l'orchestrazione dei container nel cloud computing moderno. Questa piattaforma open-source automatizza il deployment, la scalabilità e la gestione delle applicazioni containerizzate. Invece di gestire manualmente i singoli server, gli sviluppatori descrivono lo stato desiderato dell'infrastruttura e Kubernetes si occupa di mantenerlo, gestendo automaticamente i failover e il bilanciamento del carico. Sebbene la sua curva di apprendimento sia ripida, offre una flessibilità ineguagliabile per le architetture a microservizi distribuite.",
            "summary": "Kubernetes automatizza la gestione dei container nel cloud, garantendo scalabilità e affidabilità per architetture a microservizi, pur essendo complesso da apprendere."
        },
        {
            "text": "Il modello di sicurezza Zero Trust sta rimpiazzando i tradizionali approcci basati sul perimetro di rete, ormai obsoleti nell'era del lavoro remoto. Il principio fondamentale è 'non fidarsi mai, verificare sempre': ogni richiesta di accesso, sia interna che esterna alla rete aziendale, deve essere autenticata, autorizzata e crittografata prima di essere concessa. Questo approccio minimizza i rischi di movimenti laterali da parte di attaccanti che sono riusciti a violare le difese esterne, garantendo una protezione granulare per dati e applicazioni critiche.",
            "summary": "Il modello Zero Trust supera la sicurezza perimetrale richiedendo verifica continua per ogni accesso, proteggendo così i dati da minacce interne ed esterne."
        },
        {
            "text": "L'adozione della metodologia DevOps mira ad abbattere i silos tra i team di sviluppo software (Dev) e quelli operativi (Ops). Attraverso pratiche come la Continuous Integration e Continuous Delivery (CI/CD), l'automazione dei test e il monitoraggio continuo, le aziende possono rilasciare aggiornamenti software più frequentemente e con maggiore stabilità. Questo cambio culturale, oltre che tecnico, favorisce una collaborazione più stretta, riduce il time-to-market e migliora la qualità complessiva del prodotto finale grazie a feedback rapidi e iterativi.",
            "summary": "DevOps unisce sviluppo e operazioni tramite automazione e CI/CD, accelerando i rilasci software, migliorando la qualità e favorendo la collaborazione."
        },
        {
            "text": "Il calcolo quantistico rappresenta un paradigma informatico rivoluzionario che sfrutta le leggi della meccanica quantistica. A differenza dei computer classici che usano bit (0 o 1), i computer quantistici utilizzano qubit, che possono esistere in una sovrapposizione di stati. Questa capacità permette di eseguire calcoli paralleli massivi, offrendo potenzialità enormi per la crittografia, la scoperta di nuovi farmaci e l'ottimizzazione di sistemi complessi, sebbene la stabilità dei qubit rimanga una sfida tecnologica critica.",
            "summary": "Il calcolo quantistico usa i qubit per calcoli paralleli superiori, promettendo rivoluzioni in crittografia e medicina, nonostante le sfide di stabilità."
        },
        {
            "text": "La tecnologia blockchain funge da registro digitale distribuito, immutabile e trasparente. Ogni transazione viene registrata in un 'blocco' crittografato e collegato indissolubilmente al precedente, rendendo impossibile la manomissione dei dati senza il consenso della rete. Oltre alle criptovalute come il Bitcoin, la blockchain trova applicazione negli smart contract, nella tracciabilità della filiera alimentare e nella gestione sicura dell'identità digitale, eliminando la necessità di intermediari centralizzati di fiducia.",
            "summary": "La blockchain è un registro distribuito sicuro e immutabile utile per criptovalute, smart contract e tracciabilità, eliminando intermediari centrali."
        },
        {
            "text": "Gli attacchi ransomware sono diventati una delle minacce informatiche più diffuse e dannose per le aziende moderne. In questo tipo di attacco, un malware cifra i dati della vittima, rendendoli inaccessibili fino al pagamento di un riscatto, solitamente richiesto in criptovaluta. Le varianti più aggressive utilizzano la tecnica della 'doppia estorsione', minacciando non solo di bloccare i dati ma anche di pubblicarli online se il pagamento non viene effettuato, causando enormi danni reputazionali e legali.",
            "summary": "Il ransomware cifra i dati aziendali chiedendo un riscatto per il ripristino; la doppia estorsione minaccia anche la pubblicazione dei dati rubati."
        },
        {
            "text": "Le reti 5G segnano un salto generazionale rispetto al 4G, offrendo velocità di trasmissione dati drasticamente superiori e una latenza quasi nulla. Questa tecnologia non serve solo a navigare più velocemente da smartphone, ma abilita l'Internet of Things (IoT) su vasta scala, la guida autonoma e la chirurgia remota. Grazie alla capacità di connettere milioni di dispositivi per chilometro quadrato, il 5G costituisce l'infrastruttura spina dorsale per le smart city del futuro e l'industria 4.0.",
            "summary": "Il 5G offre alta velocità e bassa latenza, abilitando IoT, guida autonoma e smart city, connettendo massicciamente dispositivi e infrastrutture."
        }
    ]

def evaluate_summarization(num_samples=3, model_name="efederici/it5-base-summarization"):
    """
    Evaluates the summarization model using ROUGE metrics on a curated dataset.
    """
    logger.info("Initializing evaluation...")
    
    # 1. Load Metric
    try:
        rouge = evaluate.load("rouge")
    except Exception as e:
        logger.error(f"Failed to load ROUGE metric: {e}")
        return

    # 2. Load Dataset (Manual)
    logger.info("Loading evaluation dataset (Manual Curated Samples)...")
    full_data = get_manual_test_data()
    data_subset = full_data[:num_samples]

    # 3. Load Model
    logger.info(f"Loading model: {model_name}")
    try:
        summarizer = SummarizerModule(model_name=model_name)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    predictions = []
    references = []
    results_table = []

    logger.info(f"Starting generation for {len(data_subset)} samples...")
    
    for i, item in enumerate(data_subset):
        article = item['text']
        reference_summary = item['summary']
        
        try:
            # Simplified call matching the SummarizerModule signature
            generated_summary = summarizer.summarize(article)
        except Exception as e:
            logger.error(f"Error summarizing sample {i}: {e}")
            continue

        predictions.append(generated_summary)
        references.append(reference_summary)
        
        results_table.append([
            f"Sample {i+1}", 
            generated_summary, 
            reference_summary
        ])
        
        logger.info(f"Processed {i+1}/{len(data_subset)} samples")

    # 4. Compute Metrics
    if not predictions:
        logger.error("No predictions generated.")
        return

    logger.info("Computing ROUGE scores...")
    results = rouge.compute(predictions=predictions, references=references)
    
    # 5. Display Results
    print("\n" + "="*50)
    print("QUALITATIVE COMPARISON")
    print("="*50)
    print(tabulate(results_table, headers=["ID", "Generated Summary", "Reference (Gold)"], tablefmt="grid"))
    
    print("\n" + "="*50)
    print("QUANTITATIVE EVALUATION (ROUGE SCORES)")
    print("="*50)
    
    metrics_display = [
        ["ROUGE-1 (Unigram overlap)", f"{results['rouge1']*100:.2f}%"],
        ["ROUGE-2 (Bigram overlap)", f"{results['rouge2']*100:.2f}%"],
        ["ROUGE-L (Longest Common Subsequence)", f"{results['rougeL']*100:.2f}%"]
    ]
    
    print(tabulate(metrics_display, headers=["Metric", "Score"], tablefmt="simple"))
    
    print("\nInterpretation:")
    print("- ROUGE-1 > 40% is generally considered good.")
    print("- ROUGE-2 correlates well with fluency.")
    
    return results

if __name__ == "__main__":
    samples = 3
    if len(sys.argv) > 1:
        try:
            samples = int(sys.argv[1])
        except ValueError:
            pass
            
    evaluate_summarization(num_samples=samples)