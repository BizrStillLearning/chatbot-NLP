import json
import joblib
import random

with open('data/intents.json', 'r') as file:
    data = json.load(file)

model = joblib.load('models/model.pkl')

print("Bot siap! Ketik 'keluar' untuk berhenti.")

while True:
    pesan = input("Kamu: ")
    if pesan.lower() == 'keluar':
        print("Bot: Sampai jumpa!")
        break

    tebakan_tag = model.predict([pesan])[0]

    for intent in data['intents']:
        if intent['tag'] == tebakan_tag:
            jawaban = random.choice(intent['responses'])
            print(f"Bot: {jawaban}")
            break