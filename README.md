#  Credit Card Fraud Detection

A Machine Learning project that detects fraudulent credit card transactions using multiple classification algorithms. This project compares the performance of Logistic Regression, Decision Tree, and Random Forest to identify fraudulent transactions.

---

##  Project Overview

Credit card fraud has become a major concern in the digital payment industry. The goal of this project is to build a machine learning model capable of classifying transactions as either:

-  Legitimate Transaction
-  Fraudulent Transaction

The project trains multiple machine learning models and compares their performance to determine the most effective algorithm.

---

##  Algorithms Used

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

---

##  Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

##  Dataset

This project uses the Credit Card Fraud Detection dataset containing transaction information such as:

- Merchant
- Category
- Transaction Amount
- Gender
- City
- State
- Latitude & Longitude
- Merchant Location
- Customer Information
- Fraud Label (`is_fraud`)

### Dataset Files

- `fraudTrain.csv`
- `fraudTest.csv`

> **Note:** The dataset files are **not included in this GitHub repository** because they exceed GitHub's file size limit. Please download them separately and place them in the project folder before running the program.

Folder structure:

```
CreditCardFraudDetection/
│
├── fraud_detection.py
├── fraudTrain.csv
├── fraudTest.csv
└── README.md
```

---

## ⚙️ Installation

Install the required libraries:

```bash
python -m pip install pandas numpy matplotlib scikit-learn
```

---

##  How to Run

Run the following command:

```bash
python fraud_detection.py
```

---

##  Features

- Loads training and testing datasets
- Data preprocessing
- Converts categorical data into numerical values
- Trains multiple machine learning models
- Evaluates model performance
- Displays confusion matrix
- Generates model comparison graph
- Predicts fraudulent transactions

---

## 📈 Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 📷 Sample Output

```
==========================================
MODEL COMPARISON
==========================================

Model                     Accuracy

Logistic Regression       0.9961
Decision Tree             0.9752
Random Forest             ...
```

---

##  Future Improvements

- Feature Engineering
- Hyperparameter Tuning
- SMOTE for handling class imbalance
- Cross Validation
- ROC Curve Visualization
- Precision-Recall Curve

---

##  Author

**ajit069**

Machine Learning Internship Project (CodSoft)

---

##  If you found this project useful

Please consider giving the repository a ⭐ on GitHub.
