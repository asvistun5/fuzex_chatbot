import telebot, random, os
from dotenv import load_dotenv
from model.src.model import Model

load_dotenv()
bot = telebot.TeleBot(os.getenv("TOKEN"))

emojis = ["🤣", "😎", "🤔", "🔥", "🙃", "😦", "😧", "🫤", "😴", "🤙"]
model = Model("./model/ru.json")

def is_emoji(text):
    return all("\U0001F000" <= ch <= "\U0001FAFF" for ch in text)

@bot.message_handler(commands=['start', 'help'])
def start(message):
    bot.send_message(message.chat.id, "Привет! Напиши сообщение.")

@bot.message_handler(content_types=['sticker'])
def sticker_handler(message):
    bot.send_message(message.chat.id, random.choice(emojis))

@bot.message_handler(func=lambda m: is_emoji(m.text))
def emoji_handler(message):
    bot.send_message(message.chat.id, random.choice(emojis))

@bot.message_handler(content_types=['text'])
def text_handler(message):
    response = model.get_response(message.text)
    bot.send_message(message.chat.id, response)

bot.infinity_polling()