# 📨 Spam SMS Detection in Telugu

This repository contains a Machine Learning and Deep Learning based system to detect whether an SMS message written in **Telugu** is *Spam* or *Ham (Not Spam)*. The project focuses on applying **Natural Language Processing (NLP)** techniques to a regional language dataset where prebuilt solutions are limited.

---

## 🧠 Project Overview

Spam SMS messages are a major problem and can include fraudulent or harmful content. While spam detection is well explored for English, regional languages like **Telugu** lack robust solutions.

This project:
- Preprocesses Telugu text
- Trains Machine Learning and Deep Learning models
- Evaluates performance using standard metrics
- Provides a backend + frontend interface to test messages

---

## 🛠️ Technologies Used

- **Programming Language:** Python  
- **Libraries:**  
  - pandas, numpy  
  - scikit-learn  
  - TensorFlow / PyTorch  
  - Transformers (Hugging Face)  
- **NLP Techniques:**  
  - Text cleaning  
  - Tokenization  
  - TF-IDF / Embeddings  
- **Deployment:**  
  - Backend: FastAPI  
  - Frontend: Streamlit  

---

## ⚙️ Methodology

### 1. Data Collection
- Telugu SMS datasets are stored in Excel format.
- Messages are labeled as `Spam` or `Ham`.

### 2. Data Preprocessing
- Removal of noise and unwanted characters
- Tokenization of Telugu text
- Conversion into numerical representations

### 3. Model Training
- Traditional ML models (Naive Bayes, SVM, etc.)
- Deep Learning models using Transformers

### 4. Model Evaluation
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### 5. Application Integration
- Backend API serves trained models
- Frontend allows users to test custom SMS messages

---

## 📈 Results

- Model performance metrics and plots are available in the `Results_Images/` folder
- Comparisons between ML and DL models are shown in notebooks
