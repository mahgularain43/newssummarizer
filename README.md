# 📰 News Summarizer – LexRank vs BART (Transformers)

This project benchmarks **traditional extractive summarization** (LexRank) against a **state-of-the-art transformer model** (BART-large-CNN) using the CNN/DailyMail dataset.  
A full evaluation is performed using **ROUGE-1, ROUGE-2, ROUGE-L** scores.

---

## 🚀 Features
- LexRank summarizer (Sumy)
- BART Transformer summarizer
- ROUGE evaluation script
- Automatic dataset loading (HuggingFace)
- Exported scores & summaries

---

## 📦 Requirements

transformers
datasets
sumy
rouge-score
nltk
tqdm

Install via:

pip install -r requirements.txt

yaml
Copy code

---

## ▶️ Run the project

python main.py

yaml
Copy code

Outputs will be saved in:

results/rouge_scores.csv
results/sample_summaries.json

yaml
Copy code

---

## 📊 Example ROUGE Output

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|------|---------|---------|---------|
| LexRank | 0.28 | 0.10 | 0.21 |
| BART | 0.44 | 0.22 | 0.40 |

---

## 📚 Dataset
CNN/DailyMail (3.0.0)  
Loaded directly:

```python
from datasets import load_dataset
load_dataset("cnn_dailymail", "3.0.0")
📁 Repository Contents
main.py → runs LexRank + BART + ROUGE

utils.py → helper functions

results/ → summary outputs

notebooks/ → Jupyter notebook with analysis

