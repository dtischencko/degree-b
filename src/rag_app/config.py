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
CUDA_NUMBER = os.getenv('CUDA_NUMBER')
RETRIEVER_MODEL_NAME = os.getenv('RETRIEVER_MODEL_NAME')
RERANKER_MODEL_NAME = os.getenv('RERANKER_MODEL_NAME')
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME')
LLM_DTYPE = torch.bfloat16 if os.getenv('LLM_DTYPE') == 'bfloat' else torch.float16
MODEL_CACHE = os.getenv('MODEL_CACHE')

os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_NUMBER
DEVICE = 'cuda' if cuda.is_available() else 'cpu'

S_PROMPT = os.getenv('S_PROMPT')

logging.info(f"CUDA_NUMBER: {CUDA_NUMBER}; DEVICE: {DEVICE}")
