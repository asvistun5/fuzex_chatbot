# import bot
from src.model import Model

model = Model("model/datasets/ru.json")
run = True

while run:
    inp = input("> ")
    
    if inp.lower() in ["exit", "quit", "выход", "стоп"]:
        run = False

    print(model.get_response(inp))
    print(model.get_sentiment(inp))