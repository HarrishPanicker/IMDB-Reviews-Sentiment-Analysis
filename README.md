# Social Media Computing: IMDB Movie Reviews Sentiment & Aspect-Based Sentiment Analysis

A comprehensive comparative study on sentiment analysis and Aspect-Based Sentiment Analysis (ABSA) utilizing a 100k-row subset of the Kaggle IMDB Movie Review dataset[cite: 2, 7].

---

## Project Overview

This project investigates automated sentiment classification and aspect extraction from movie reviews to understand what drives audience reception[cite: 2, 3]. It implements and compares multiple approaches ranging from lexicon baselines and traditional machine learning algorithms to transformer-based deep learning models[cite: 2].

* **Dataset:** 100,000 row sample from the Kaggle IMDB Movie Reviews dataset (Enam Biswas)[cite: 7].
* **Task:** 3-class sentiment classification (Positive, Neutral, Negative) and Aspect-Based Sentiment Analysis (ABSA)[cite: 6, 7].
* **Author Contributions:** 
  * **Model Training:** Led the fine-tuning and optimization pipelines across traditional machine learning models and deep learning transformers (DistilBERT), configuring training hyper-parameters (such as learning rates, epochs, and seed locking)[cite: 11, 15].
  * **Insight Derivation:** Conducted evaluation, comparative performance analysis, confusion matrix interpretation, and aspect-based sentiment proportion breakdowns[cite: 12, 13, 17, 18].

---

## Methodology & Pipeline

The project follows a structured multi-phase pipeline[cite: 6]:

1. **Phase 1: Data Preparation**
   * Handled null values and duplicate removal[cite: 6].
   * Executed a 7-step text preprocessing pipeline and target engineering for 3-class categorization[cite: 6].
2. **Phase 2: Traditional Machine Learning & Lexicon Models**
   * **VADER:** Lexicon-based baseline[cite: 6, 8].
   * **Logistic Regression & Support Vector Machine (SVM):** Trained on a 5,000-feature TF-IDF matrix[cite: 6, 8].
   * **Random Forest:** Evaluated for baseline performance comparison[cite: 8].
3. **Phase 3: Deep Learning & ABSA**
   * **DistilBERT Fine-Tuning:** Implemented using PyTorch, Hugging Face `transformers`, `datasets`, and `accelerate` libraries[cite: 10, 11]. Configured with a learning rate of $3e-5$, batch size of 32, and 4 epochs with deterministic seed locking[cite: 11].
   * **Aspect-Based Sentiment Analysis (ABSA):** Combined `spaCy` for aspect extraction with `VADER` for sentiment polarity scoring over extracted terms[cite: 4, 6].

---

## Comparative Results

### Sentiment Classification Performance
Among the tested models, **DistilBERT** achieved the highest overall accuracy and macro F1-score, particularly excelling at detecting neutral text where lexicon baselines struggled[cite: 15, 18].

| Model | Accuracy | Negative F1 | Neutral F1 | Positive F1 | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **VADER (Lexicon)**[cite: 15] | 66.58%[cite: 15] | 0.45[cite: 15] | 0.02[cite: 15] | 0.79[cite: 15] | 0.42[cite: 15] |
| **Logistic Regression**[cite: 15] | 79.50%[cite: 15] | 0.74[cite: 15] | 0.26[cite: 15] | 0.89[cite: 15] | 0.63[cite: 15] |
| **SVM**[cite: 15] | 79.59%[cite: 15] | 0.74[cite: 15] | 0.16[cite: 15] | 0.88[cite: 15] | 0.60[cite: 15] |
| **Random Forest**[cite: 15] | 74.60%[cite: 15] | 0.59[cite: 15] | <0.01[cite: 15] | 0.84[cite: 15] | 0.48[cite: 15] |
| **DistilBERT**[cite: 15] | **79.80%**[cite: 15] | **0.76**[cite: 15] | **0.39**[cite: 15] | **0.89**[cite: 15] | **0.68**[cite: 15] |

### Aspect-Based Sentiment Analysis (ABSA) Insights
* **Top Mentioned Aspects:** The most frequently discussed elements in movie reviews center around terms like *film/movie*, *story*, *characters*, *scenes*, and *people*[cite: 16, 19].
* **Sentiment Breakdown:** Positive sentiment overwhelmingly dominates across nearly all core aspects, though neutral sentiment presents challenges due to ambiguity in natural language text[cite: 17, 18].

---

## Discussion & Key Findings

* **Model Trade-offs:** While DistilBERT offers superior accuracy and handles neutral sentiments better, it is computationally expensive (~30–45 minutes on GPU). Logistic Regression and SVM offer highly competitive performance (~79.5% accuracy) with significantly faster execution times (~5 minutes on CPU), making them viable alternatives depending on resource constraints[cite: 15, 18, 19].
* **Dataset Bias:** The dataset exhibits a heavy positive class skew, reflecting general reviewer tendencies on platforms like IMDb[cite: 18, 19].

---

## Group Contributors

* **Leong Kah Hang** (1221305975)[cite: 1]
* **Harrish Panicker** (1221308940)[cite: 1] — *Model Training & Insight Derivation*
