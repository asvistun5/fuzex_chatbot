from json import load
from .strings import normalize


class JsonParser:
    def __init__(self, *paths):

        self.patterns = []
        self.tags = []
        self.responses = {}
        self.errors = []
        self.sentiments = {}


        for path in paths:
            self.load_file(path)


    def load_file(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = load(f)

        if isinstance(data, list):
            for item in data:
                tag = item.get("tag")

                if tag == "error":
                    self.errors = item.get("res", [])
                    continue

                self.responses[tag] = item.get("res", [])

                for p in item.get("pt", []):
                    self.patterns.append(normalize(p))
                    self.tags.append(tag)