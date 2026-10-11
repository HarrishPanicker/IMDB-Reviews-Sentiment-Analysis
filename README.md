# Social Media Computing: IMDB Movie Reviews Sentiment & Aspect-Based Sentiment Analysis

A comprehensive comparative study on sentiment analysis and Aspect-Based Sentiment Analysis (ABSA) utilizing a 100k-row subset of the Kaggle IMDB Movie Review dataset.

---

## Project Overview

This project investigates automated sentiment classification and aspect extraction from movie reviews to understand what drives audience reception. It implements and compares multiple approaches ranging from lexicon baselines and traditional machine learning algorithms to transformer-based deep learning models.

* **Dataset:** 100,000 row sample from the Kaggle IMDB Movie Reviews dataset (Enam Biswas).
* **Task:** 3-class sentiment classification (Positive, Neutral, Negative) and Aspect-Based Sentiment Analysis (ABSA).
* **Author Contributions:** 
  * **Model Training:** Led the fine-tuning and optimization pipelines across traditional machine learning models and deep learning transformers (DistilBERT), configuring training hyper-parameters (such as learning rates, epochs, and seed locking).
  * **Insight Derivation:** Conducted evaluation, comparative performance analysis, confusion matrix interpretation, and aspect-based sentiment proportion breakdowns.

---

## Methodology & Pipeline

The project follows a structured multi-phase pipeline:

1. **Phase 1: Data Preparation**
   * Handled null values and duplicate removal.
   * Executed a 7-step text preprocessing pipeline and target engineering for 3-class categorization.
2. **Phase 2: Traditional Machine Learning & Lexicon Models**
   * **VADER:** Lexicon-based baseline.
   * **Logistic Regression & Support Vector Machine (SVM):** Trained on a 5,000-feature TF-IDF matrix.
   * **Random Forest:** Evaluated for baseline performance comparison.
3. **Phase 3: Deep Learning & ABSA**
   * **DistilBERT Fine-Tuning:** Implemented using PyTorch, Hugging Face `transformers`, `datasets`, and `accelerate` libraries. Configured with a learning rate of $3e-5$, batch size of 32, and 4 epochs with deterministic seed locking.
   * **Aspect-Based Sentiment Analysis (ABSA):** Combined `spaCy` for aspect extraction with `VADER` for sentiment polarity scoring over extracted terms.

---

## Comparative Results

### Sentiment Classification Performance
Among the tested models, **DistilBERT** achieved the highest overall accuracy and macro F1-score, particularly excelling at detecting neutral text where lexicon baselines struggled.

| Model | Accuracy | Negative F1 | Neutral F1 | Positive F1 | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **VADER (Lexicon)** | 66.58% | 0.45 | 0.02 | 0.79 | 0.42 |
| **Logistic Regression** | 79.50% | 0.74 | 0.26 | 0.89 | 0.63 |
| **SVM** | 79.59% | 0.74 | 0.16 | 0.88 | 0.60 |
| **Random Forest** | 74.60% | 0.59 | <0.01 | 0.84 | 0.48 |
| **DistilBERT** | **79.80%** | **0.76** | **0.39** | **0.89** | **0.68** |

### Aspect-Based Sentiment Analysis (ABSA) Insights
* **Top Mentioned Aspects:** The most frequently discussed elements in movie reviews center around terms like *film/movie*, *story*, *characters*, *scenes*, and *people*.
* **Sentiment Breakdown:** Positive sentiment overwhelmingly dominates across nearly all core aspects, though neutral sentiment presents challenges due to ambiguity in natural language text.

---

## Discussion & Key Findings

* **Model Trade-offs:** While DistilBERT offers superior accuracy and handles neutral sentiments better, it is computationally expensive (~30–45 minutes on GPU). Logistic Regression and SVM offer highly competitive performance (~79.5% accuracy) with significantly faster execution times (~5 minutes on CPU), making them viable alternatives depending on resource constraints.
* **Dataset Bias:** The dataset exhibits a heavy positive class skew, reflecting general reviewer tendencies on platforms like IMDb.

---

## Group Contributors

* **Leong Kah Hang** (1221305975)
* **Harrish Panicker** (1221308940) — *Model Training & Insight Derivation*
