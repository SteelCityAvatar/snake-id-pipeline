# 🐍 Reddit Snake Identifier Pipeline

This project is an automated pipeline that scrapes image posts from the `r/whatsthissnake` subreddit, classifies the snake in each image using GPT-4o, and evaluates the classification results using Google's Gemini model as a sanity check. It also generates a PDF report for visual review.

---

## 🔧 Features

- 🔎 Scrapes snake-related Reddit posts (images only)
- 🧠 Classifies snake species via OpenAI's GPT-4o
- 📜 Extracts "ground truth" from Reddit user replies using Gemini LLM
- 📈 Evaluates predictions with precision, recall, and accuracy
- 📂 Outputs results to CSV + generates a visual PDF report

---

## 📁 Folder Structure

```
.
├── images/                  # Downloaded Reddit images
├── results/
│   └── classification_results.csv
├── output.pdf               # PDF report
├── snekid.py                # Main pipeline script
```

---

## 🛠️ Requirements

- Python 3.9+
- [OpenAI API key](https://platform.openai.com/account/api-keys)
- [Gemini API key](https://makersuite.google.com/app)
- Reddit app credentials (for PRAW)

Install required packages:

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Set these in your environment (e.g., `.env`, bash profile, etc.):


```bash
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_app_name

OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
```

---

## 🚀 How to Run

```bash
python run.py
```

1. Scrapes image posts from r/whatsthissnake
2. Classifies the snakes using GPT-4o
3. Extracts human-suggested labels using Gemini
4. Generates `output.pdf` and `classification_results.csv`
5. Evaluates performance using Gemini feedback

---

## 📊 Evaluation Metrics

The script calculates:
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **Accuracy** = TP / (TP + FP + FN)

Gemini serves as the evaluator for comparison between predicted and ground truth labels.

---

## 📸 PDF Report

The `output.pdf` includes:
- Snake image
- GPT-4o prediction
- Top & Reliable Reddit comments
- Extracted ground truth

---

## 📄 License

MIT License

---

## ✨ Future Ideas

- Fine-tune snake classification models
- Use CLIP or BLIP2 for fallback visual classification
- Build a UI using Streamlit or Gradio
