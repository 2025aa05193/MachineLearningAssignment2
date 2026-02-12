import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from MBTIClass import MBTIType
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
    ,ConfusionMatrixDisplay
)

st.title("MBTI Personality Classification App")
@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# -----------------------------
# Model Selection
# -----------------------------
model_option = st.selectbox(
    "Select Model",
    (
        "Select a Model",
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    )
)

# Load model
model_dict = {
    "Logistic Regression": "model/trained_models/logistic_regression.pkl",
    "Decision Tree": "model/trained_models/DecisionTree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

#Initialize Selection to Null
model = None

if model_option != "Select a Model":
    model_path = model_dict[model_option]
    model = load_model(model_path)

                        
# -----------------------------
# Upload Test Data
# -----------------------------
uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file is not None and model is not None:
    df = pd.read_csv(uploaded_file)

    # Separate features and label
    X_test = df.drop(columns=["Response Id","Personality"])
    y_test = df["Personality"]

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    #y_pred = [MBTIType(i).name for i in y_pred]
    #y_test = [MBTIType(i).name for i in y_test]

    # -----------------------------
    # Metrics
    # -----------------------------
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    mcc = matthews_corrcoef(y_test, y_pred)

    st.subheader("Evaluation Metrics")

    metrics_dict = {
        "Metric": [
            "Accuracy",
            "AUC (Macro)",
            "Precision (Macro)",
            "Recall (Macro)",
            "F1 Score (Macro)",
            "MCC Score"
        ],
        "Score": [
            accuracy,
            auc,
            precision,
            recall,
            f1,
            mcc
        ]
    }

    metrics_df = pd.DataFrame(metrics_dict)

    # Round values
    metrics_df["Score"] = metrics_df["Score"].round(4)

    st.dataframe(metrics_df, use_container_width=True)


    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    st.subheader("Confusion Matrix")
    class_names = [e.name for e in MBTIType]
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(10, 10))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(ax=ax, xticks_rotation=90)
    plt.title("Confusion Matrix")

    st.pyplot(fig)

    # -----------------------------
    # Classification Report
    # -----------------------------
    st.subheader("Classification Report")
    report = classification_report(y_pred, y_pred)
    st.text(report)
