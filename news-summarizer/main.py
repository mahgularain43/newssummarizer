import os
import json
import csv

from datasets import load_dataset
from transformers import pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from rouge_score import rouge_scorer
from tqdm import tqdm
import nltk

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
# Force Transformers to ignore TensorFlow and only use PyTorch
os.environ["TRANSFORMERS_NO_TF"] = "1"

NUM_SAMPLES = 50          # you can increase for stronger eval
TARGET_SENTENCES = 3
MODEL_NAME = "facebook/bart-large-cnn"


# -------------------------------------------------------------------
# NLTK setup
# -------------------------------------------------------------------
def setup_nltk():
    """
    Downloads required NLTK models if not present.
    """
    nltk.download("punkt", quiet=True)
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        # older NLTK versions don't have punkt_tab, ignore
        pass


# -------------------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------------------
def load_data(num_samples: int = NUM_SAMPLES):
    """
    Load CNN/DailyMail dataset from Hugging Face and take a subset.
    Returns:
        articles (List[str]), summaries (List[str])
    """
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")
    dataset = dataset.select(range(num_samples))
    return dataset["article"], dataset["highlights"]


# -------------------------------------------------------------------
# SUMMARIZERS
# -------------------------------------------------------------------
def lexrank_summary(text: str, target_sentences: int = TARGET_SENTENCES) -> str:
    """
    Extractive summarization using LexRank (Sumy).
    """
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, target_sentences)
    return " ".join(str(s) for s in summary)


def transformer_summarizer():
    """
    Abstractive summarization using BART-large-CNN via transformers pipeline.
    """
    return pipeline("summarization", model=MODEL_NAME, tokenizer=MODEL_NAME)


# -------------------------------------------------------------------
# METRICS
# -------------------------------------------------------------------
def compute_rouge(refs, preds):
    """
    Compute average ROUGE-1, ROUGE-2, ROUGE-L F1 scores.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for ref, pred in zip(refs, preds):
        sc = scorer.score(ref, pred)
        scores["rouge1"].append(sc["rouge1"].fmeasure)
        scores["rouge2"].append(sc["rouge2"].fmeasure)
        scores["rougeL"].append(sc["rougeL"].fmeasure)

    return {k: sum(v) / len(v) for k, v in scores.items()}


# -------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------
def main():
    setup_nltk()

    print("📥 Loading dataset...")
    texts, refs = load_data()
    print(f"✅ Loaded {len(texts)} samples.")

    print("⚙️ Initialising transformer summarizer...")
    hf_sum = transformer_summarizer()

    lex_summaries = []
    bart_summaries = []
    samples = []

    print("📝 Generating summaries...")
    for i, text in tqdm(enumerate(texts), total=len(texts), desc="Summarizing"):
        # LexRank summary (extractive)
        lex_summary = lexrank_summary(text)
        lex_summaries.append(lex_summary)

        # BART summary (abstractive)
        bart_out = hf_sum(
            text[:2000],          # truncate long articles
            max_length=120,
            min_length=30,
            do_sample=False
        )[0]["summary_text"]
        bart_summaries.append(bart_out)

        samples.append({
            "id": i,
            "article_snippet": text[:500] + "...",
            "lexrank_summary": lex_summary,
            "bart_summary": bart_out,
            "reference_summary": refs[i]
        })

    print("📊 Computing ROUGE scores...")
    lex_rouge = compute_rouge(refs, lex_summaries)
    bart_rouge = compute_rouge(refs, bart_summaries)

    os.makedirs("results", exist_ok=True)

    # Save ROUGE scores
    rouge_path = os.path.join("results", "rouge_scores.csv")
    with open(rouge_path, "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "rouge1", "rouge2", "rougeL"])
        writer.writerow(["LexRank", lex_rouge["rouge1"], lex_rouge["rouge2"], lex_rouge["rougeL"]])
        writer.writerow(["BART",    bart_rouge["rouge1"], bart_rouge["rouge2"], bart_rouge["rougeL"]])

    # Save qualitative examples
    samples_path = os.path.join("results", "sample_summaries.json")
    with open(samples_path, "w", encoding="utf8") as f:
        json.dump(samples, f, indent=2)

    print("✅ Done.")
    print(f"📁 ROUGE saved to: {rouge_path}")
    print(f"📁 Sample summaries saved to: {samples_path}")
    print("\nAverage ROUGE:")
    print("LexRank:", lex_rouge)
    print("BART   :", bart_rouge)


if __name__ == "__main__":
    main()
