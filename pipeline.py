# src/pipeline.py
import os, csv
from typing import List, Tuple

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
SRC_DIR = os.path.join(BASE_DIR, "src")
PROMPT_PATH = os.path.join(SRC_DIR, "prompt.txt")
#PROMPT_PATH = os.path.join(BASE_DIR, "tests", "prompts", "prompt8.txt")
CSV_OUT = os.path.join(BASE_DIR, "metrics", "results.csv")

AUDIO_DIRS: List[Tuple[str, int]] = [
    (os.path.join(DATA_DIR, "harmful"), 1),
    (os.path.join(DATA_DIR, "safe"), 0),
]

from src.transcript import stt  
from src.classification import label_conversation     

def evaluate():
    """
    Run the end-to-end evaluation over labeled audio directories.
    Invokes the speech-to-text function stt(), and then the label_conversation() function to label the transcribed audio.
    Appends results to `results.csv`, and prints Precision, Recall, and F1.
    """
    with open(PROMPT_PATH, "r", encoding="utf-8") as _pf:
        _prompt_text = _pf.read()
    y_true_all, y_pred_all = [], []

    # prepare CSV
    os.makedirs(os.path.dirname(CSV_OUT) or ".", exist_ok=True)
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "y_true", "y_pred", "reason"])

        for folder, y_true in AUDIO_DIRS:
            if not os.path.isdir(folder):
                continue
            for name in sorted(os.listdir(folder)):
                if not name.lower().endswith(".mp3"):
                    continue
                path = os.path.join(folder, name)

                # 1) ASR → dialogue-style transcript
                structured = stt(path)

                # 2) LLM classifier
                label, reason, _ = label_conversation(structured, _prompt_text)
                y_pred = 1 if label == 1 else 0

                writer.writerow([path, y_true, y_pred, reason])
                y_true_all.append(y_true)
                y_pred_all.append(y_pred)
                print(f"{name}: gt={y_true} pred={y_pred} | label: {label} reason: {reason}")

    # --- metrics ---
    tp = sum(1 for y, p in zip(y_true_all, y_pred_all) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(y_true_all, y_pred_all) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(y_true_all, y_pred_all) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true_all, y_pred_all) if y == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    print("\n=== Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    # --- save metrics to text file ---
    os.makedirs("metrics", exist_ok=True)
    with open(os.path.join("metrics", "metrics.txt"), "a", encoding="utf-8") as f:
        f.write("\n=== Evaluation Summary ===\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-score:  {f1:.4f}\n")
    print("Saved metrics summary to metrics/metrics.txt")

    print(f"\nSaved: {CSV_OUT}")

if __name__ == "__main__":
    evaluate()