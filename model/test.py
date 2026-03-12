# import bot
from src.model import Model
# 7109112907:AAGRvfodaaQLOnH7-zkQtVTuJp_gxBZ4zQU
model = Model("./ru.json")

while True:
    inp = input("> ")
    if inp.lower() in ["выход", "exit", "quit"]:
        break
    print(model.get_response(inp))
    print(model.get_sentiment(inp))