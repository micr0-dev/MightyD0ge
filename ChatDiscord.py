from pickle import NONE
import discord
from dotenv import load_dotenv
import os
import time

#Constants
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

guild = None

client = discord.Client()



import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NONE
device = NONE

@client.event
async def on_ready():
    global model
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
    print("Connected")

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    sentence = message.content

    msg = ""
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                msg = random.choice(intent['responses'])
    if msg != "":
        time.sleep(len(message.content)/200)
        async with message.channel.typing():
            time.sleep(len(msg)/120)
        await message.channel.send(msg)

client.run(TOKEN)