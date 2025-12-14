# Phishing Email Detection

A comprehensive machine learning project for identifying phishing emails using Natural Language Processing (NLP) and deep learning techniques.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Table of Contents

- [Overview](#overview)
- [Project Workflow](#project-workflow)
- [Key Features](#key-features)
- [Technologies](#technologies)
- [Project Structure](#project-structure)
- [Analysis Components](#analysis-components)
- [Models](#models)
- [Results](#results)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## Overview

This project implements a multi-faceted approach to phishing email detection, combining linguistic analysis, topic modeling, sentiment analysis, and machine learning classification. Developed as part of the Social Media Analysis course, the system employs state-of-the-art NLP techniques to identify malicious emails with high accuracy.

## Project Workflow

The analysis pipeline consists of several interconnected stages:

1. **Linguistic Analysis**: POS tagging and Named Entity Recognition
2. **Topic Modeling**: LDA, Guided LDA, and BERTopic analysis
3. **Sentiment Analysis**: Lexicon-based, BERT, and LLM-based sentiment evaluation
4. **Feature Engineering**: TF-IDF vectorization and text embeddings
5. **Classification**: Random Forest and fine-tuned RoBERTa models
6. **Model Interpretability**: SHAP value analysis

## Key Features

- **Multi-Model Approach**: Combines traditional ML (Random Forest) with transformer-based models (RoBERTa)
- **Comprehensive NLP Analysis**: POS tagging, NER, and topic modeling to understand phishing patterns
- **Advanced Sentiment Analysis**: Multiple sentiment analysis techniques including LLM-based evaluation using Gemma via Langchain
- **Interpretable Results**: SHAP values provide insights into model decision-making
- **Topic Discovery**: Identifies common phishing themes (fake software offers, financial scams, pharmaceutical spam, lottery schemes)

## Technologies

### Core Frameworks
- **Python 3.8+**
- **PyTorch**: Deep learning framework for model training
- **Transformers (Hugging Face)**: Pre-trained language models and fine-tuning
- **Scikit-learn**: Traditional ML algorithms and evaluation metrics

### NLP & Text Processing
- **NLTK**: Natural language processing and tokenization
- **Stanza/CoreNLP**: Advanced NER and POS tagging
- **Gensim**: Topic modeling (LDA)
- **GuidedLDA**: Seed-guided topic modeling
- **BERTopic**: BERT-based topic modeling

### Machine Learning & Analysis
- **SHAP**: Model interpretability and feature importance
- **Langchain**: LLM integration for sentiment analysis
- **Sentence-Transformers**: Text embeddings

### Visualization
- **Matplotlib**: General plotting
- **Seaborn**: Statistical visualizations
- **pyLDAvis**: Interactive topic model visualization
- **WordCloud**: Text frequency visualization

## Project Structure

```
Phising_Email_Detection/
│
├── README.md                                  # Project documentation
│
├── POS.ipynb                                  # Part-of-Speech tagging analysis
├── NER.ipynb                                  # Named Entity Recognition
├── Freq.ipynb                                 # Frequency analysis
│
├── topic.ipynb                                # Topic modeling (LDA, GuidedLDA, BERTopic)
│
├── lexiconsentiment_phishing.ipynb           # Lexicon-based sentiment (phishing emails)
├── lexiconsentiment_safe.ipynb               # Lexicon-based sentiment (safe emails)
├── bert_sentiment.ipynb                       # BERT sentiment analysis
├── LLM_sentiment.ipynb                        # LLM-based sentiment analysis (Gemma)
│
├── Phishing_Email_Detection_tfidf.ipynb      # Random Forest with TF-IDF features
├── phishing_email_detection_bert.ipynb       # RoBERTa fine-tuning and classification
└── phishing-email-detection-bert.ipynb       # Alternative BERT implementation
```

## Analysis Components

### 1. Linguistic Analysis

**Part-of-Speech (POS) Tagging** ([POS.ipynb](POS.ipynb))
- Identifies grammatical patterns characteristic of phishing emails
- Reveals frequent use of imperative verbs, urgency markers, and monetary terms

**Named Entity Recognition (NER)** ([NER.ipynb](NER.ipynb))
- Extracts key entities: organizations, locations, monetary values, dates
- Identifies commonly impersonated organizations (Microsoft, financial institutions)
- Reveals geographic patterns in phishing attempts

### 2. Topic Modeling

**Standard LDA** ([topic.ipynb](topic.ipynb))
- Discovers latent topics in phishing emails
- Optimal topic count determined via perplexity and coherence metrics

**Guided LDA**
- Incorporates domain knowledge with seed words
- Topics identified: free software, financial investment, pharmaceuticals, money transfer schemes, HTML/spam content

**BERTopic**
- Context-aware topic modeling using BERT embeddings
- Provides more granular topic separation
- Identifies specialized phishing categories (battery scams, religious fraud, academic scams)

### 3. Sentiment Analysis

**Lexicon-Based Analysis** ([lexiconsentiment_phishing.ipynb](lexiconsentiment_phishing.ipynb), [lexiconsentiment_safe.ipynb](lexiconsentiment_safe.ipynb))
- Comparative sentiment analysis between phishing and legitimate emails
- Uses established sentiment lexicons

**BERT Sentiment** ([bert_sentiment.ipynb](bert_sentiment.ipynb))
- Fine-grained sentiment classification using BERT
- Captures nuanced emotional manipulation in phishing emails

**LLM Sentiment** ([LLM_sentiment.ipynb](LLM_sentiment.ipynb))
- Leverages Gemma LLM via Langchain
- Provides contextual sentiment understanding

### 4. Classification Models

**Random Forest with TF-IDF** ([Phishing_Email_Detection_tfidf.ipynb](Phishing_Email_Detection_tfidf.ipynb))
- Traditional feature extraction using TF-IDF vectorization
- Random Forest classifier with hyperparameter tuning
- Baseline performance comparison

**Fine-Tuned RoBERTa** ([phishing_email_detection_bert.ipynb](phishing_email_detection_bert.ipynb))
- Transfer learning using pre-trained RoBERTa
- Fine-tuning on phishing email dataset
- Superior performance with contextual understanding

**Model Interpretability**
- SHAP (SHapley Additive exPlanations) analysis
- Identifies key features driving classification decisions
- Provides transparency in model predictions

## Models

### Random Forest Classifier
- **Features**: TF-IDF vectors, custom embeddings
- **Purpose**: Baseline comparison, feature importance analysis
- **Advantages**: Fast training, interpretable feature weights

### RoBERTa (Robustly Optimized BERT)
- **Architecture**: Transformer-based language model
- **Training**: Fine-tuned on phishing email dataset
- **Advantages**: Contextual understanding, state-of-the-art performance
- **Interpretability**: Enhanced with SHAP value analysis

## Results

The project successfully identifies several characteristic patterns of phishing emails:

**Common Phishing Themes**:
- Free software and discount offers
- Financial investment opportunities
- Pharmaceutical spam (viagra, cialis, herbal products)
- Nigerian/advance-fee fraud schemes
- Cryptocurrency and lottery scams

**Key Indicators**:
- Frequent use of terms: "free", "click", "money", "offer", "limited time"
- High occurrence of URLs and HTML content
- Impersonation of legitimate organizations
- Urgency markers and time-limited offers
- Unusual entity mentions (foreign countries, unfamiliar organizations)

## Dataset

**Source**: [Phishing Emails Dataset on Kaggle](https://www.kaggle.com/datasets/subhajournal/phishingemails/data)

**Description**:
- Contains both phishing and legitimate emails
- Labeled dataset for supervised learning
- Diverse phishing attack types

**Preprocessing**:
- Sentence tokenization
- Punctuation and special character removal
- Stop word filtering
- Lemmatization

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 16GB+ RAM recommended for large model training
- GPU recommended for BERT fine-tuning

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/Phising_Email_Detection.git
cd Phising_Email_Detection
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install torch torchvision
pip install transformers
pip install scikit-learn
pip install nltk
pip install stanza
pip install gensim
pip install guidedlda
pip install bertopic
pip install langchain
pip install shap
pip install matplotlib seaborn
pip install wordcloud
pip install pyLDAvis
pip install jupyter
```

4. **Download NLTK data**:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

5. **Download CoreNLP** (for NER):
```python
import stanza
stanza.install_corenlp()
```

## Usage

### Running Individual Analyses

1. **Linguistic Analysis**:
```bash
jupyter notebook POS.ipynb  # Part-of-Speech tagging
jupyter notebook NER.ipynb  # Named Entity Recognition
```

2. **Topic Modeling**:
```bash
jupyter notebook topic.ipynb
```

3. **Sentiment Analysis**:
```bash
jupyter notebook LLM_sentiment.ipynb
```

4. **Classification**:
```bash
jupyter notebook phishing_email_detection_bert.ipynb
```

### End-to-End Pipeline

Run notebooks in the following order for complete analysis:
1. Data preprocessing and frequency analysis ([Freq.ipynb](Freq.ipynb))
2. Linguistic analysis ([POS.ipynb](POS.ipynb), [NER.ipynb](NER.ipynb))
3. Topic modeling ([topic.ipynb](topic.ipynb))
4. Sentiment analysis (lexicon → BERT → LLM)
5. Model training and evaluation (TF-IDF → RoBERTa)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

- **Course**: Social Media Analysis
- **Dataset**: [SubhaJournal on Kaggle](https://www.kaggle.com/datasets/subhajournal/phishingemails/data)
- **Video Presentation**: [YouTube](https://www.youtube.com/watch?v=bQb1AmYcOvA)
- **Libraries**: Hugging Face Transformers, NLTK, Gensim, and the entire Python data science community

---

**Note**: This project is for educational and research purposes. Always exercise caution with email security in real-world applications.
