import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

stemmer = StemmerFactory().create_stemmer()
stopword = StopWordRemoverFactory().create_stop_word_remover()

with open('data/intents.json', 'r') as file:
    data = json.load(file)

kalimat = []
label = []

print("Mulai memproses teks (Membersihkan kata & mencari kata dasar)... ini butuh beberapa detik.")

for intent in data['intents']:
    for pattern in intent['patterns']:
        teks_bersih = stopword.remove(pattern)
        teks_dasar = stemmer.stem(teks_bersih)

        kalimat.append(teks_dasar)
        label.append(intent['tag'])

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(kalimat, label)

joblib.dump(model, 'models/model.pkl')
print("✅ Model TF-IDF yang lebih cerdas berhasil dilatih dan disimpan!")