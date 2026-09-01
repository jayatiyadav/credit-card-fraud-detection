import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="FraudShield",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD MODEL & DATA
# =========================================================

@st.cache_resource
def load_model():
    model = joblib.load("models/fraud_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler


@st.cache_data
def load_data():
    return pd.read_csv("data/creditcard.csv")


model, scaler = load_model()
df = load_data()


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.main-title {
    font-size: 46px;
    font-weight: 800;
    margin-bottom: 0;
}

.subtitle {
    font-size: 18px;
    color: #9aa0a6;
    margin-top: 4px;
}

.card {
    padding: 20px;
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,0.25);
    background: rgba(128,128,128,0.05);
}

.result-fraud {
    padding: 25px;
    border-radius: 15px;
    border: 1px solid rgba(255,70,70,0.5);
    background: rgba(255,70,70,0.08);
    text-align: center;
}

.result-safe {
    padding: 25px;
    border-radius: 15px;
    border: 1px solid rgba(50,200,120,0.5);
    background: rgba(50,200,120,0.08);
    text-align: center;
}

.small-text {
    color: #9aa0a6;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## 💳 FraudShield")
st.sidebar.caption("AI-Powered Fraud Detection")

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📊 Analytics",
        "🤖 Model Performance",
        "🔍 Fraud Detection"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Random Forest classifier trained using SMOTE "
    "to handle severe class imbalance."
)


# =========================================================
# COMMON DATA
# =========================================================

total_transactions = len(df)

fraud_transactions = int(
    df["Class"].sum()
)

legitimate_transactions = (
    total_transactions - fraud_transactions
)

fraud_rate = (
    fraud_transactions / total_transactions
) * 100


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">💳 FraudShield</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-Powered Credit Card Fraud Detection System'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.header("🏠 Dashboard")

    # ---------------- Metrics ----------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "💳 Total Transactions",
        f"{total_transactions:,}"
    )

    col2.metric(
        "🟢 Legitimate",
        f"{legitimate_transactions:,}"
    )

    col3.metric(
        "🔴 Fraud",
        f"{fraud_transactions:,}"
    )

    col4.metric(
        "⚠️ Fraud Rate",
        f"{fraud_rate:.2f}%"
    )

    st.divider()

    # ---------------- Charts ----------------

    st.subheader("📈 Transaction Overview")

    col1, col2 = st.columns(2)

    with col1:

        fig, ax = plt.subplots()

        counts = df["Class"].value_counts()

        ax.bar(
            ["Legitimate", "Fraud"],
            [
                counts.get(0, 0),
                counts.get(1, 0)
            ]
        )

        ax.set_title(
            "Fraud vs Legitimate Transactions"
        )

        ax.set_ylabel("Number of Transactions")

        st.pyplot(fig)

        plt.close(fig)

    with col2:

        fig, ax = plt.subplots()

        ax.hist(
            df["Amount"],
            bins=50
        )

        ax.set_title(
            "Transaction Amount Distribution"
        )

        ax.set_xlabel("Transaction Amount")

        ax.set_ylabel("Frequency")

        st.pyplot(fig)

        plt.close(fig)

    st.divider()

    # ---------------- Model Summary ----------------

    st.subheader("🤖 Model Summary")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="card">
            <h3>🌲 Random Forest</h3>
            <p class="small-text">
            Ensemble classification algorithm
            used for fraud detection.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="card">
            <h3>⚖️ SMOTE</h3>
            <p class="small-text">
            Synthetic Minority Over-sampling
            Technique used for class imbalance.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="card">
            <h3>📊 ROC-AUC</h3>
            <p class="small-text">
            Model ROC-AUC: 0.9664
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.info(
        "Credit card fraud is a highly imbalanced classification "
        "problem. Therefore, precision, recall, F1-score and "
        "ROC-AUC are important evaluation metrics."
    )


# =========================================================
# ANALYTICS
# =========================================================

elif page == "📊 Analytics":

    st.markdown(
        '<div class="main-title">📊 Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Exploratory analysis of transaction behaviour'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # ---------------- Amount Analysis ----------------

    st.subheader("💰 Transaction Amount Analysis")

    col1, col2 = st.columns(2)

    legitimate = df[df["Class"] == 0]["Amount"]
    fraud = df[df["Class"] == 1]["Amount"]

    with col1:

        fig, ax = plt.subplots()

        ax.boxplot(
            [legitimate, fraud],
            tick_labels=[
                "Legitimate",
                "Fraud"
            ]
        )

        ax.set_title(
            "Transaction Amount Comparison"
        )

        ax.set_ylabel("Amount")

        st.pyplot(fig)

        plt.close(fig)

    with col2:

        fig, ax = plt.subplots()

        ax.hist(
            fraud,
            bins=30
        )

        ax.set_title(
            "Fraud Transaction Amount Distribution"
        )

        ax.set_xlabel("Amount")

        ax.set_ylabel("Frequency")

        st.pyplot(fig)

        plt.close(fig)

    st.divider()

    # ---------------- Fraud Statistics ----------------

    st.subheader("🔴 Fraud Transaction Statistics")

    fraud_stats = fraud.describe()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Average",
        f"${fraud.mean():.2f}"
    )

    col2.metric(
        "Median",
        f"${fraud.median():.2f}"
    )

    col3.metric(
        "Minimum",
        f"${fraud.min():.2f}"
    )

    col4.metric(
        "Maximum",
        f"${fraud.max():.2f}"
    )

    st.divider()

    # ---------------- Correlation ----------------

    st.subheader("🔗 Features Most Correlated With Fraud")

    correlations = (
        df.corr(numeric_only=True)["Class"]
        .drop("Class")
        .abs()
        .sort_values(ascending=False)
        .head(10)
    )

    st.bar_chart(correlations)

    st.divider()

    st.subheader("🔎 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "🤖 Model Performance":

    st.markdown(
        '<div class="main-title">🤖 Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Random Forest evaluation on the test dataset'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("📈 Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "ROC-AUC",
        "96.64%"
    )

    col2.metric(
        "Fraud Precision",
        "92%"
    )

    col3.metric(
        "Fraud Recall",
        "74%"
    )

    col4.metric(
        "Fraud F1-Score",
        "82%"
    )

    st.divider()

    st.subheader("🎯 Confusion Matrix")

    # Results from trained Random Forest model

    confusion = np.array([
        [56645, 6],
        [25, 70]
    ])

    fig, ax = plt.subplots()

    image = ax.imshow(confusion)

    ax.set_title(
        "Random Forest Confusion Matrix"
    )

    ax.set_xlabel("Predicted Label")

    ax.set_ylabel("Actual Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["Legitimate", "Fraud"]
    )

    ax.set_yticklabels(
        ["Legitimate", "Fraud"]
    )

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                confusion[i, j],
                ha="center",
                va="center"
            )

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    st.subheader("📋 Classification Summary")

    performance = pd.DataFrame({
        "Metric": [
            "Legitimate Precision",
            "Legitimate Recall",
            "Legitimate F1",
            "Fraud Precision",
            "Fraud Recall",
            "Fraud F1",
            "ROC-AUC"
        ],
        "Score": [
            "1.00",
            "1.00",
            "1.00",
            "0.92",
            "0.74",
            "0.82",
            "0.9664"
        ]
    })

    st.dataframe(
        performance,
        use_container_width=True,
        hide_index=True
    )

    st.success(
        "Random Forest performed better than Logistic Regression "
        "on the test dataset, especially in fraud precision."
    )


# =========================================================
# FRAUD DETECTION
# =========================================================

elif page == "🔍 Fraud Detection":

    st.markdown(
        '<div class="main-title">🔍 Fraud Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Use the trained Random Forest model to detect potentially '
        'fraudulent transactions'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # =====================================================
    # DEMO TRANSACTION
    # =====================================================

    st.subheader("🧪 Quick Demo")

    st.write(
        "Test the trained model using a transaction from "
        "the dataset."
    )

    if st.button("▶️ Run Demo Transaction"):

        demo_row = df.sample(
            1,
            random_state=42
        )

        required_features = [
            "Time",
            "V1", "V2", "V3", "V4", "V5",
            "V6", "V7", "V8", "V9", "V10",
            "V11", "V12", "V13", "V14", "V15",
            "V16", "V17", "V18", "V19", "V20",
            "V21", "V22", "V23", "V24", "V25",
            "V26", "V27", "V28",
            "Amount"
        ]

        X_demo = demo_row[required_features]

        X_demo_scaled = scaler.transform(
            X_demo
        )

        prediction = model.predict(
            X_demo_scaled
        )[0]

        probability = model.predict_proba(
            X_demo_scaled
        )[0][1]

        st.write("### Transaction")

        st.dataframe(
            demo_row[
                ["Time", "Amount", "Class"]
            ],
            use_container_width=True,
            hide_index=True
        )

        if prediction == 1:

            st.markdown(
                f"""
                <div class="result-fraud">
                <h2>🚨 Potential Fraud Detected</h2>
                <h3>Fraud Probability: {probability * 100:.2f}%</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="result-safe">
                <h2>✅ Transaction Appears Legitimate</h2>
                <h3>Fraud Probability: {probability * 100:.2f}%</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    # =====================================================
    # CSV UPLOAD
    # =====================================================

    st.subheader("📂 Upload Transaction CSV")

    uploaded_file = st.file_uploader(
        "Upload a CSV file containing transaction data",
        type=["csv"]
    )

    if uploaded_file is not None:

        uploaded_df = pd.read_csv(
            uploaded_file
        )

        st.success(
            "CSV uploaded successfully!"
        )

        st.subheader("📄 Uploaded Data")

        st.dataframe(
            uploaded_df.head(),
            use_container_width=True
        )

        required_features = [
            "Time",
            "V1", "V2", "V3", "V4", "V5",
            "V6", "V7", "V8", "V9", "V10",
            "V11", "V12", "V13", "V14", "V15",
            "V16", "V17", "V18", "V19", "V20",
            "V21", "V22", "V23", "V24", "V25",
            "V26", "V27", "V28",
            "Amount"
        ]

        missing_columns = [
            column
            for column in required_features
            if column not in uploaded_df.columns
        ]

        if missing_columns:

            st.error(
                "❌ Required columns are missing."
            )

            st.write(
                missing_columns
            )

        else:

            if st.button(
                "🚨 Detect Fraud",
                type="primary"
            ):

                X_new = uploaded_df[
                    required_features
                ]

                X_scaled = scaler.transform(
                    X_new
                )

                predictions = model.predict(
                    X_scaled
                )

                probabilities = (
                    model.predict_proba(
                        X_scaled
                    )[:, 1]
                )

                results = uploaded_df.copy()

                results["Prediction"] = np.where(
                    predictions == 1,
                    "Fraud",
                    "Legitimate"
                )

                results["Fraud Probability (%)"] = (
                    probabilities * 100
                ).round(2)

                st.subheader(
                    "📋 Prediction Results"
                )

                st.dataframe(
                    results,
                    use_container_width=True
                )

                fraud_count = int(
                    (predictions == 1).sum()
                )

                legitimate_count = int(
                    (predictions == 0).sum()
                )

                st.divider()

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "🔴 Potential Fraud",
                    fraud_count
                )

                col2.metric(
                    "🟢 Legitimate",
                    legitimate_count
                )

                col3.metric(
                    "📊 Total",
                    len(predictions)
                )

                # ---------------- Probability Chart ----------------

                st.subheader(
                    "📈 Fraud Probability"
                )

                probability_df = pd.DataFrame({
                    "Transaction": range(
                        1,
                        len(probabilities) + 1
                    ),
                    "Fraud Probability": (
                        probabilities * 100
                    )
                })

                st.bar_chart(
                    probability_df.set_index(
                        "Transaction"
                    )
                )

                # ---------------- Download ----------------

                csv = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "⬇️ Download Predictions",
                    csv,
                    "fraud_predictions.csv",
                    "text/csv"
                )

    st.divider()

    st.caption(
        "⚠️ This system provides machine-learning based risk "
        "predictions and should not be treated as a definitive "
        "determination of fraud."
    )