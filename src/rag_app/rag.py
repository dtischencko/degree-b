import os
import logging
import pickle
from config import DEVICE, RETRIEVER_MODEL_NAME, RERANKER_MODEL_NAME, LLM_MODEL_NAME, MODEL_CACHE, LLM_DTYPE, S_PROMPT

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import CrossEncoder, SentenceTransformer


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)


class RAGHandler:

    def __init__(
            self,
            retriever_model_name=RETRIEVER_MODEL_NAME,
            reranker_model_name=RERANKER_MODEL_NAME,
            llm_model_name=LLM_MODEL_NAME,
            cache_dir=MODEL_CACHE,
        ) -> None:
        assert retriever_model_name is not None or reranker_model_name is not None or llm_model_name is not None, f"Error type of some model: {retriever_model_name}, {reranker_model_name}, {llm_model_name}, please set all env RETRIEVER_MODEL_NAME, RERANKER_MODEL_NAME, LLM_MODEL_NAME or pass it as an attribute!"

        if cache_dir is None:
            cache_dir = os.path.abspath('./hf_cache')
            logging.warning(f"Cache directory is None => cache_dir is set by default ({cache_dir})")

        self.llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=LLM_DTYPE,
            cache_dir=cache_dir,
        ).to(DEVICE)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_NAME,
            cache_dir=cache_dir,
        )
        self.retriever_model = SentenceTransformer(
            RETRIEVER_MODEL_NAME, 
            cache_folder=cache_dir
        )
        self.reranker_model = CrossEncoder(
            RERANKER_MODEL_NAME, 
            max_length=512, 
            automodel_args={'cache_dir': cache_dir}, 
            tokenizer_args={'cache_dir': cache_dir}
        )


    def do_rag(
            self, 
            query: str,
            threshold_confidience: float,
        ) -> dict:

        is_discard = False
        candidates = self.retrieve_and_rerank(query)

        top_hit_id = candidates[0]['corpus_id']
        top_hit_score = candidates[0]['score']

        if top_hit_score < threshold_confidience:
            is_discard = True

        answer = self.generate(query, self.corpus[top_hit_id], is_discard)

        return {
            "answer": answer,
        }


    def generate(
        self,
        query: str,
        candidate: str,
        is_discard: bool = False,
    ) -> str:
        
        messages = [
            {"role": "system", "content": S_PROMPT},
            {"role": "user", "content": query},
        ]

        logging.debug(f"MESSAGES:\n{messages}")

        input_ids = self.llm_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.llm_model.device)

        terminators = [
            self.llm_tokenizer.eos_token_id,
            self.llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.llm_model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        response = outputs[0][input_ids.shape[-1]:]
        decoded_response = self.llm_tokenizer.decode(response, skip_special_tokens=True)

        logging.info(f"ANSWER (is_discard:{is_discard})\n{decoded_response}")

        return decoded_response


    def retrieve_and_rerank(  
        self,
        query: str,
        need_retrieve=False
    ) -> list:

        results = self.reranker_model.rank(
            query,
            self.corpus,
            top_k=10,
        )

        logging.info(f"SEMANTIC SEARCH RESULTS:\n{results}")

        return results
