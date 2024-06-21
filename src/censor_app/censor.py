import os
import logging
import random
import pickle
from config import RETRIEVER_MODEL_NAME, DOMAIN_TOPIC, MODEL_CACHE

from sentence_transformers import SentenceTransformer, util


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)


class Censor:

    def __init__(
            self,
            domain=DOMAIN_TOPIC,
            retriever_model_name=RETRIEVER_MODEL_NAME,
            cache_dir=MODEL_CACHE,
        ) -> None:
        assert retriever_model_name is not None, f"Error type of some model: {retriever_model_name}, please set env RETRIEVER_MODEL_NAME or pass it as an attribute!"

        if cache_dir is None:
            cache_dir = os.path.abspath('./hf_cache')
            logging.warning(f"Cache directory is None => cache_dir is set by default ({cache_dir})")

        self.censor_model = SentenceTransformer(
            retriever_model_name, 
            cache_folder=cache_dir
        )
        self.domain = domain


    def do_filt(
            self, 
            query: str,
            threshold: float
        ):

        context_vector = self.get_context_vector(query)
        domain_vector = self.censor_model.encode(self.domain, convert_to_tensor=True)
        cosine_similarity = util.cos_sim(domain_vector, context_vector)
        logging.info(f"COSINE_SIMILARITY: {cosine_similarity}({threshold})")

        return str(cosine_similarity.item()) if cosine_similarity > threshold else ''


    def get_context_vector(self, input_str, permutation_times=100):
        permuted_texts = [permute_words(input_str) for _ in range(permutation_times)]
        enc_texts = self.censor_model.encode(permuted_texts, convert_to_tensor=True)
        enc_context = enc_texts.mean(dim=0)

        return enc_context


def permute_words(input_str, n_permutes=32):
    words = input_str.split()
    n_grams = len(words) // 3
    num_words = len(words)
    
    # Проверка на корректность входных данных
    if num_words < n_grams:
        logging.warning("Ошибка: количество слов в строке меньше, чем требуемое число слов для перестановки")
        return ""

    permutated_sentence = input_str
    if n_grams > 1:
        for i in range(n_permutes):
            if i > 0:
                words = permutated_sentence.split()
            # Выбор случайного индекса начала подстроки
            start_index = random.randint(0, num_words - n_grams)
            end_index = start_index + n_grams
            # Перемешивание выбранных слов
            subset = words[start_index:end_index]
            random.shuffle(subset)
            words[start_index:end_index] = subset
            permutated_sentence = ' '.join(words)
    else:
        for i in range(n_permutes):
            words_copy = words[:]
            random.shuffle(words_copy)
            permutated_sentence = ' '.join(words_copy)
    
    return permutated_sentence
