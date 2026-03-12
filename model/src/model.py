import json, random, re, pymorphy3, numpy as np
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz


class Model:
    def __init__(self, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.morph = pymorphy3.MorphAnalyzer()
        self.patterns = []
        self.tags = []
        self.responses = {}
        self.err_res = ["Извини, я не понимаю тебя.", "Можешь переформулировать?", "Я не уверен, что ты имеешь в виду.", "Поясни, пожалуйста."]
        
        self.embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        self.memory = []
        self.memory_size = 10
        self.memory_embeddings = []

        self.self_memory = []
        self.self_memory_size = 8

        for item in self.data:
            tag = item.get("tag")

            if tag == "err":
                self.err_res = item.get("res", self.err_res)
                continue

            self.responses[tag] = item.get("res", [])

            for p in item.get("pt", []):
                self.patterns.append(self.normalize(p))
                self.tags.append(tag)

        self.pattern_embeddings = self.embedder.encode(self.patterns)

    def normalize(self, text):
        text = re.sub(r"[^\w\s]", "", text.lower())
        words = text.split()
        lemmas = [self.morph.parse(w)[0].normal_form for w in words]
        return " ".join(lemmas)
    
    def cosine(self, text):
        emb = self.embedder.encode([text])[0]

        sims = np.dot(self.pattern_embeddings, emb) / (
            np.linalg.norm(self.pattern_embeddings, axis=1) * np.linalg.norm(emb)
        )

        idx = np.argmax(sims)
        score = sims[idx]

        return idx, score
    
    def build_context(self, input):
        emb = self.embedder.encode([input])[0]

        self.memory.append(input)
        self.memory_embeddings.append(emb)

        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
            self.memory_embeddings.pop(0)
    
    def get_sentiment(self, text):
        text_norm = self.normalize(text)
        words = set(text_norm.split())

        positive = {"хорошо", "отлично", "прекрасно", "замечательно", "круто", "супер"}
        negative = {"плохо", "ужасно", "отвратительно", "хреново", "хуёво"}

        pos_score = len(words & positive)
        neg_score = len(words & negative)

        letters = [c for c in text if c.isalpha()]

        if letters:
            caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        else:
            caps_ratio = 0

        if caps_ratio > 0.5:
            neg_score += 1

        if pos_score > 0 and pos_score >= neg_score:
            return "positive"
        elif neg_score > 0:
            return "negative"
        else:
            return "neutral"

    def get_response(self, user_input):
        self.build_context(user_input)

        text = self.normalize(user_input)
        idx, score = self.cosine(text)

        if score < 0.35:
            best_score = 0
            best_idx = -1
            for i, pattern in enumerate(self.patterns):
                fscore = fuzz.ratio(text, pattern)
                if fscore > best_score:
                    best_score = fscore
                    best_idx = i
            if best_score > 85:
                tag = self.tags[best_idx]
            else:
                tag = None
        else:
            tag = self.tags[idx]

        if tag:
            possible_res = [r for r in self.responses[tag] if r not in self.self_memory]
            if not possible_res:
                possible_res = self.responses[tag]
            response = random.choice(possible_res)
            self.self_memory.append(response)
            if len(self.self_memory) > self.self_memory_size:
                self.self_memory.pop(0)
            return response
        else:
            return random.choice(self.err_res)