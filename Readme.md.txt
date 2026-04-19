# 💬 Tweet Sentiment Analyzer

A NLP project that classifies tweets as Positive or Negative using a 
Twitter-trained RoBERTa model, deployed as an interactive web app.

🔗 **Live Demo**: https://huggingface.co/spaces/Abhishek3411/tweet-sentiment-analyzer

## 📌 Project Overview
This project compares a rule-based TextBlob baseline against a pretrained 
RoBERTa transformer model fine-tuned on 58 million tweets.

## 📊 Model Comparison
| Model | Accuracy | Negative F1 |
|-------|----------|-------------|
| TextBlob (baseline) | 82% | 0.17 |
| RoBERTa (Twitter) | 79% | 0.32 |

## 🛠️ Tech Stack
- Python, Pandas, Streamlit
- HuggingFace Transformers (RoBERTa)
- TextBlob, Scikit-learn
- Deployed on HuggingFace Spaces

## 👨‍💻 Author
Abhishek N Salian | MSc Data Science, University of Europe for Applied Sciences, Berlin
