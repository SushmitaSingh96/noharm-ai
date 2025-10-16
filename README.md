# Speech Harm Classification Pipeline

This repository contains a local, privacy-preserving pipeline for detecting and classifying harmful content in audio conversations. It features two main modules: audio transcription and harm classification, both located in the `src/` directory. Evaluation results (Precision, Recall, F1, and latency) are saved in the `metrics/metrics.txt` file inside the `metrics/` folder. 

## 📂 Project Structure
```
noharm-ai/
│
├── src/
│   ├── transcript.py       # Converts audio clips to text using Faster Whisper
│   ├── classification.py   # Classifies the text using a quantized Llama model
│
├── pipeline.py             # Evaluates model performance (Precision, Recall, F1)
├── ui_app.py               # Streamlit interface for testing new audio samples
├── data/                   # Folder for evaluation audio files (harmful/safe)
├── metrics/                # Contains evaluation results stored in metrics.txt
├── models/
└── README.md
```

## 🧩 Module Descriptions
- **`transcript.py`**: Converts audio clips to structured text using a transcription model (Faster Whisper).
- **`classification.py`**: Takes transcribed text and uses a quantized Llama model to classify as:
  - `0` → Non-harmful
  - `1` → Harmful
  Also returns a short reason explaining the decision.

## 🧠 Model Setup
Before running the project, create a folder named `models/` in the root directory and place the quantized model file `Llama-3.2-3B-Instruct-Q4_K_M.gguf` inside it.  
You can download the model from [The Hugging Face repository](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/blob/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf).  
This model is required for text classification and is not automatically downloaded.

## ⚙️ Running the Pipeline
1. Create a `data/` folder in the root directory with:
   ```
   data/
   ├── harmful/   # Harmful audio samples
   └── safe/      # Non-harmful audio samples
   ```
2. Run the evaluation script:

   ```bash
   python pipeline.py
   ```
   - Transcribes and classifies each audio file
   - Saves results in a CSV file
   - Displays Precision, Recall, and F1 Score in the console

> **Note:** The original dataset provided for this task is **not included in the repository** due to privacy restrictions. To reproduce evaluation results, create a `data/` folder in the root directory using the same structure as the provided dataset (with `harmful/` and `safe/` subfolders) before running the pipeline.

## 🎧 Testing on New Audio
1. Run the Streamlit UI:
   ```bash
   streamlit run ui_app.py
   ```
2. Upload an audio file
3. The system returns:
   ```json
   { "label": 0 or 1, "reason": "<short explanation>" }
   ```

## 🧠 Summary
- Audio → Text → Harm/Non-Harm classification
- Uses quantized Llama 3.2 3B for efficient CPU inference
- Outputs both classification label and reason
- Supports batch evaluation (`pipeline.py`) and interactive testing (`ui_app.py`)

## 🧰 Requirements
- Python ≥ 3.9  
- faster-whisper  
- llama-cpp-python  
- streamlit  
- psutil, re, json, csv
