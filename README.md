# Social Media Analysis Final Project
## Phising Email Analysis & Detection

---
### Description
This project, developed for the Social Media Analysis course, focuses on detecting phishing attempts in emails using Python. Our approach began with Part-of-Speech (POS) tagging and Named Entity Recognition (NER) to identify linguistic patterns characteristic of phishing emails. We then applied three topic modeling techniques—LDA, Guided LDA, and BERTopic—to uncover common themes in these emails.

For sentiment analysis, we employed the LLM Gemma via Langchain to assess the emotional tone of the emails. Following extensive data exploration, we used embedding techniques to train a Random Forest model and fine-tuned a RoBERTa model to classify emails as either phishing or legitimate. To enhance the interpretability of our models, we utilized Shapley Values to understand the key features that led to an email being classified as phishing.

---
### Built With

- Python
- PyTorch
- Transformers
- Scikit-learn
- Langchain
- NLTK
- Matplotlib
- Seaborn

---
### Data Source
Source link：https://www.kaggle.com/datasets/subhajournal/phishingemails/data

---
### Video
Youtube link：https://www.youtube.com/watch?v=bQb1AmYcOvA
