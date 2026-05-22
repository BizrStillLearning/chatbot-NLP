import json
import joblib
import random
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

stemmer = StemmerFactory().create_stemmer()
stopword = StopWordRemoverFactory().create_stop_word_remover()

with open('data/intents.json', 'r') as file:
    data = json.load(file)
model = joblib.load('models/model.pkl')

console.print(Panel("[bold green]Bot NLP Siap![/bold green]\nKetik [bold red]'keluar'[/bold red] untuk berhenti.",
                    title="🤖 AI Chatbot", expand=False))

while True:
    pesan = Prompt.ask("\n[bold blue]Kamu[/bold blue]")

    if pesan.lower() == 'keluar':
        console.print(Panel("Sampai jumpa lagi!", title="🤖 Bot", style="green", expand=False))
        break

    teks_bersih = stopword.remove(pesan)
    teks_dasar = stemmer.stem(teks_bersih)

    console.print(f"[dim italic yellow] -> Bot memproses teks: '{teks_dasar}'[/dim italic yellow]")

    probabilitas = model.predict_proba([teks_dasar])[0]
    tebakan_tag = model.classes_[probabilitas.argmax()]
    nilai_keyakinan = probabilitas.max()

    console.print(
        f"[dim italic yellow] -> Bot yakin {nilai_keyakinan * 100:.1f}% ini adalah kategori '{tebakan_tag}'[/dim italic yellow]")

    if nilai_keyakinan < 0.3:
        jawaban = "Maaf, saya belum mengerti maksudmu. Bisa gunakan kalimat yang lebih jelas?"
    else:
        for intent in data['intents']:
            if intent['tag'] == tebakan_tag:
                jawaban = random.choice(intent['responses'])
                break

    console.print(Panel(jawaban, title="🤖 Bot", style="green", border_style="green", expand=False))