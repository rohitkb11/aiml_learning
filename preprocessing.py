
# Assignment 1: Titanic Dataset
# Data Cleaning, Visualization, Feature Scaling, and Train-Test Split

# --------------------------------------------------
# 1. IMPORTING LIBRARIES
# --------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 2. READING DATA
# --------------------------------------------------

df = pd.read_csv("Titanic-Dataset.xls")

print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nNumber of samples:", df.shape[0])
print("Number of features:", df.shape[1])


# --------------------------------------------------
# 3. DATA CLEANING
# --------------------------------------------------


# Print feature names, total values, and data types
print("\nFeature Information:")

for col in df.columns:
    print(
        f'Feature Name = "{col}",',
        f"Total Values Present = {df[col].count()},",
        f"Data Type = {df[col].dtype}"
    )

# Drop Name and PassengerId columns
df.drop(columns=["Name", "PassengerId"], inplace=True)

print("\nDataset description:")
print(df.describe())


# --------------------------------------------------
# 4. MISSING VALUE HANDLING
# --------------------------------------------------

# Detect missing values
print("\nMissing Values in Each Column:")

for col in df.columns:
    print(f"Column Name      : {col}")
    print(f"Missing Values   : {df[col].isnull().sum()}")
    print("-" * 35)

# Remove Cabin because it has too many missing values
df.drop(columns=["Cabin"], inplace=True)

# Fill missing Age values with the median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values with the mode
df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)

print("\nMissing Values After Handling:")
print(df.isnull().sum())


# --------------------------------------------------
# 5. ENCODING CATEGORICAL DATA
# --------------------------------------------------

# Encode Sex using LabelEncoder
encoder = LabelEncoder()

df["Sex"] = encoder.fit_transform(df["Sex"])

print("\nDataset after encoding Sex:")
print(df.head())

# Check unique values in Embarked
print("\nUnique values in Embarked:")
print(df["Embarked"].unique())

# One-hot encode Embarked
df = pd.get_dummies(
    df,
    columns=["Embarked"],
    dtype=int
)

print("\nDataset after one-hot encoding Embarked:")
print(df.head())


# --------------------------------------------------
# 6. DATASET SUMMARY
# --------------------------------------------------

print("\nDataset Description:")
print(df.describe())

print("\nSurvival Counts:")
print(df["Survived"].value_counts())

print("\nSurvival Rate by Passenger Class:")
print(df.groupby("Pclass")["Survived"].mean())


# --------------------------------------------------
# 7. DATA VISUALIZATION
# --------------------------------------------------

# Survival rate by passenger class and sex
plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="Pclass",
    y="Survived",
    hue="Sex"
)

plt.title("Survival Rate by Passenger Class & Sex")
plt.xlabel("Passenger Class (1st, 2nd, 3rd)")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.show()


# Boxplot for Age
plt.figure(figsize=(8, 5))

sns.boxplot(x=df["Age"])

plt.title("Boxplot of Age")
plt.xlabel("Age")
plt.tight_layout()
plt.show()


# Boxplot for Fare
plt.figure(figsize=(8, 5))

sns.boxplot(x=df["Fare"])

plt.title("Boxplot of Fare")
plt.xlabel("Fare")
plt.tight_layout()
plt.show()


# Histogram for Age
plt.figure(figsize=(8, 5))

plt.hist(df["Age"])

plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Number of People")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 8. FEATURE SCALING
# --------------------------------------------------

scaler = MinMaxScaler()

cols_to_scale = ["Age", "Fare"]

df[cols_to_scale] = scaler.fit_transform(
    df[cols_to_scale]
)

print("\nDataset after Feature Scaling:")
print(df.head(10))


# --------------------------------------------------
# 9. SEPARATING INPUT AND TARGET
# --------------------------------------------------

X = df.drop(columns=["Survived"])

y = df["Survived"]

print("\nInput Features (X):")
print(X.head())

print("\nTarget Variable (y):")
print(y.head())


# --------------------------------------------------
# 10. TRAIN-TEST SPLIT
# --------------------------------------------------

# Normal 80:20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# 11. PRINT TRAINING AND TESTING DATA INFORMATION
# --------------------------------------------------

print("\nTraining and Testing Dataset Information")

print(
    "Number of samples in training dataset:",
    X_train.shape[0]
)

print(
    "Number of features in training dataset:",
    X_train.shape[1]
)

print(
    "Number of samples in testing dataset:",
    X_test.shape[0]
)

print(
    "Number of features in testing dataset:",
    X_test.shape[1]
)

print("\nFirst 5 rows of X_train:")
print(X_train.head())

print("\nFirst 5 rows of X_test:")
print(X_test.head())

print("\nFirst 5 values of y_train:")
print(y_train.head())

print("\nFirst 5 values of y_test:")
print(y_test.head())

print("\nAssignment 1 completed successfully!")
