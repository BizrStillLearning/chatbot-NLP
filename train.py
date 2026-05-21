import json
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

with open('data/intents.json', 'r') as file:
    data = json.load(file)

kalimat = []
label = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        kalimat.append(pattern)
        label.append(intent['tag'])

model = make_pipeline(CountVectorizer(), MultinomialNB())

model.fit(kalimat, label)

joblib.dump(model, 'models/model.pkl')
print("Model berhasil dilatih dan disimpan!")