import os
import sys
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