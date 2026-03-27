import json, random, re, pymorphy3, numpy as np
import modules.strings as string

from .modules import JsonParser, sentiment
from llama_cpp import Llama

from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz


class Model:
    def __init__(self, data_path):
        
        llm = Llama.from_pretrained(
        	repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
        	filename="Phi-3-mini-4k-instruct-q4.gguf",
        )
        
        self.default_prompt = f"""
            Ты чат-бот.

            Отвечай строго по смыслу этих вариантов.
            Не придумывай новые факты.
            Можешь перефразировать, но не выходи за рамки.

            Варианты:
            <variants>

            Пользователь: <request>
            Ответ:
        """

        self.data = JsonParser(data_path)

        self.embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        self.memory = []
        self.memory_size = 10
        self.memory_embeddings = []

        self.self_memory = []
        self.self_memory_size = 8

        self.pattern_embeddings = self.embedder.encode(self.data.patterns)


    def normalize(self, text):
        return string.normalize(text)
    
    def build_context(self, input):
        emb = self.embedder.encode([input])[0]

        self.memory.append(input)
        self.memory_embeddings.append(emb)

        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
            self.memory_embeddings.pop(0)


    def similarity(self, text):
        return string.cosine(self.embedder, text, self.pattern_embeddings)
    
    def find_answer_in_data(self, idx, score, text):
        if score < 0.35:
            best_score = 0
            best_idx = -1
            for i, pattern in enumerate(self.data.patterns):
                fscore = fuzz.ratio(text, pattern)
                if fscore > best_score:
                    best_score = fscore
                    best_idx = i
            if best_score > 85:
                return self.data.tags[best_idx]
            else:
                return None
        else:
            return self.data.tags[idx]

    def gen_llm(self, prompt):
        output = self.llm(
            prompt,
            max_tokens=40,
            temperature=0.5,
            top_p=0.9,
            stop=["Пользователь:", "\n"]
        )
        return output["choices"][0]["text"].strip()
    
    
    def gen_with_template(self, user_input, responses):
        variants = "\n".join(
            f"- {r}" for r in random.sample(responses, min(3, len(responses)))
        )

        prompt = self.default_prompt \
            .replace("<variants>", variants) \
            .replace("<request>", user_input)

        return self.gen_llm(prompt)
    
    
    def get_sentiment(self, text):
        return sentiment(text)

    def get_response(self, user_input):
        self.build_context(user_input)

        text = self.normalize(user_input)
        idx, score = self.similarity(text)

        tag = self.find_answer_in_data(idx, score, text)

        if tag:
            possible_res = [r for r in self.data.responses[tag] if r not in self.self_memory]
            if not possible_res:
                possible_res = self.data.responses[tag]

            if score > 0.5:
                response = self.gen_with_template(user_input, possible_res)
            else:
                self.gen_llm('...')
        
        self.self_memory.append(response)
        
        if len(self.self_memory) > self.self_memory_size:
            self.self_memory.pop(0)
            
        return response