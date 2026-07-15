
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

print("="*50)
print(" CREDIT CARD FRAUD DETECTION SYSTEM ")
print("="*50)


print("\nLoading Training Dataset...")

train_df = pd.read_csv("fraudTrain.csv")

print("Training Dataset Loaded Successfully!")


print("\nLoading Testing Dataset...")

test_df = pd.read_csv("fraudTest.csv")

print("Testing Dataset Loaded Successfully!")


print("\nTraining Dataset Shape :", train_df.shape)
print("Testing Dataset Shape  :", test_df.shape)

print("\nTraining Dataset Columns:\n")
print(train_df.columns)

print("\nMissing Values in Training Dataset:\n")
print(train_df.isnull().sum())


plt.figure(figsize=(6,4))

train_df["is_fraud"].value_counts().plot(kind="bar")

plt.title("Fraud vs Legitimate Transactions")
plt.xlabel("0 = Legitimate     1 = Fraud")
plt.ylabel("Number of Transactions")

plt.show()


drop_columns = [
    "trans_date_trans_time",
    "cc_num",
    "first",
    "last",
    "street",
    "trans_num",
    "dob"
]

train_df.drop(columns=drop_columns, inplace=True)

test_df.drop(columns=drop_columns, inplace=True)


combined = pd.concat([train_df, test_df], ignore_index=True)

for col in combined.select_dtypes(include="object").columns:
    combined[col] = combined[col].astype("category").cat.codes

train_df = combined.iloc[:len(train_df)].copy()
test_df = combined.iloc[len(train_df):].copy()

X_train = train_df.drop("is_fraud", axis=1)

y_train = train_df["is_fraud"]

X_test = test_df.drop("is_fraud", axis=1)

y_test = test_df["is_fraud"]

print("\nData Preprocessing Completed Successfully!")


print("\n" + "="*50)
print("LOGISTIC REGRESSION")
print("="*50)

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

lr_prediction = lr.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_prediction)

print(f"\nAccuracy : {lr_accuracy:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, lr_prediction))

print("\nClassification Report:")
print(classification_report(y_test, lr_prediction))


print("\n" + "="*50)
print("DECISION TREE")
print("="*50)

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

dt_prediction = dt.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_prediction)

print(f"\nAccuracy : {dt_accuracy:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, dt_prediction))

print("\nClassification Report:")
print(classification_report(y_test, dt_prediction))


print("\n" + "="*50)
print("RANDOM FOREST")
print("="*50)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_prediction = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_prediction)

print(f"\nAccuracy : {rf_accuracy:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_prediction))

print("\nClassification Report:")
print(classification_report(y_test, rf_prediction))


results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        lr_accuracy,
        dt_accuracy,
        rf_accuracy
    ]
})

print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

print(results)


plt.figure(figsize=(8,5))

plt.bar(results["Model"], results["Accuracy"])

plt.title("Machine Learning Model Comparison")

plt.xlabel("Models")

plt.ylabel("Accuracy")

plt.ylim(0.90, 1.00)

plt.show()


best_model = results.loc[results["Accuracy"].idxmax()]

print("\n" + "="*50)
print("BEST MODEL")
print("="*50)

print(f"Model    : {best_model['Model']}")
print(f"Accuracy : {best_model['Accuracy']:.4f}")


print("\n" + "="*50)
print("SAMPLE TRANSACTION PREDICTION")
print("="*50)

sample = X_test.iloc[[0]]

prediction = rf.predict(sample)

print("\nPrediction Result:")

if prediction[0] == 1:
    print("🚨 Fraudulent Transaction")
else:
    print("✅ Legitimate Transaction")
