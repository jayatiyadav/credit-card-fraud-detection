import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# -----------------------------
# 1. Class Distribution
# -----------------------------

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Class")

plt.title("Fraud vs Legitimate Transactions")
plt.xlabel("Transaction Class")
plt.ylabel("Number of Transactions")
plt.xticks([0, 1], ["Legitimate", "Fraud"])

plt.tight_layout()
plt.show()


# -----------------------------
# 2. Transaction Amount
# -----------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Amount",
    bins=50,
    kde=True
)

plt.title("Distribution of Transaction Amount")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# -----------------------------
# 3. Amount: Fraud vs Legitimate
# -----------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Class",
    y="Amount"
)

plt.title("Transaction Amount: Fraud vs Legitimate")
plt.xlabel("Transaction Class")
plt.ylabel("Amount")
plt.xticks([0, 1], ["Legitimate", "Fraud"])

plt.tight_layout()
plt.show()


# -----------------------------
# 4. Correlation with Class
# -----------------------------

correlation = df.corr(numeric_only=True)["Class"].sort_values()

print("\nFeatures most correlated with Fraud:")
print(correlation.head(10))

print("\nFeatures most positively correlated with Fraud:")
print(correlation.tail(10))


# -----------------------------
# 5. Fraud Transactions
# -----------------------------

fraud = df[df["Class"] == 1]

print("\nFraud Transaction Statistics:")
print(fraud["Amount"].describe())