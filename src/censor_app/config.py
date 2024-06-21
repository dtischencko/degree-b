import os
import logging
import torch
from torch import cuda 
from dotenv import load_dotenv


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

load_dotenv()
RETRIEVER_MODEL_NAME = os.getenv('RETRIEVER_MODEL_NAME')
MODEL_CACHE = os.getenv('MODEL_CACHE')
DEVICE = 'cuda' if cuda.is_available() else 'cpu'
DOMAIN_TOPIC = os.getenv('DOMAIN_TOPIC')

logging.info(f"Models: {RETRIEVER_MODEL_NAME}; in cache {MODEL_CACHE}")
