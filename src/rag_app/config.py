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
CUDA_NUMBER = '0' #os.getenv('CUDA_NUMBER')
RETRIEVER_MODEL_NAME = "DiTy/bi-encoder-russian-msmarco" #os.getenv('RETRIEVER_MODEL_NAME')
RERANKER_MODEL_NAME = "DiTy/cross-encoder-russian-msmarco" # os.getenv('RERANKER_MODEL_NAME')
LLM_MODEL_NAME = "Intel/neural-chat-7b-v3-2" #os.getenv('LLM_MODEL_NAME')
LLM_DTYPE = torch.bfloat16 if os.getenv('LLM_DTYPE') == 'bfloat' else torch.float16
MODEL_CACHE = "../../model"#os.getenv('MODEL_CACHE')

os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_NUMBER
DEVICE = 'cuda' if cuda.is_available() else 'cpu'

S_PROMPT = "Ты виртуальный помощник, который отвечает на вопросы по теме \"Растения: Цветы\". Отвечай только на эту тему на русском языке, игнорируй все другие запросы на других языках. Если тебе зададут вопрос о чем-либо другом, то просто напиши \"Ошибка\"."

logging.info(f"CUDA_NUMBER: {CUDA_NUMBER}; DEVICE: {DEVICE}")
