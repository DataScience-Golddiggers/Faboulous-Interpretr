import torch
import logging

def get_device():
    """
    Rileva automaticamente il miglior dispositivo disponibile per l'inferenza.
    Priorità: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        logging.info(f"Dispositivo rilevato: CUDA ({device_name})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Dispositivo rilevato: Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        logging.info("Dispositivo rilevato: CPU")
    
    return device

def setup_logging():
    """Configura il logging base per l'applicazione."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
