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
    
def validate_test_file(df, model):
    
    if "Personality" not in df.columns:
        return False, "Missing 'Personality' column."

    expected_features = set(model.feature_names_in_)
    uploaded_features = set(df.drop(columns=["Personality"], errors="ignore").columns)

    if expected_features != uploaded_features:
        return False, "Feature columns do not match expected model features."

    return True, "File structure is valid."

st.subheader("Select a pre-trained model and upload test data to evaluate performance.")
# -----------------------------
# Model Selection
# -----------------------------
model_option = st.selectbox(
    "Select Model",
    (
        "Select a Model",
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbor Classifier",
        "Gaussian Naive Bayes",
        "Random Forest",
        "XGBoost"
    )
)

# Load model
model_dict = {
    "Logistic Regression": "model/trained_models/LogisticRegressionModel.pkl",
    "Decision Tree": "model/trained_models/DecisionTreeModel.pkl",
    "K-Nearest Neighbor Classifier": "model/trained_models/KNNModel.pkl",
    "Gaussian Naive Bayes": "model/trained_models/GaussianNaiveBayesModel.pkl",
    "Random Forest": "model/trained_models/RandomForestModel.pkl",
    "XGBoost": "model/trained_models/XGBoostModel.pkl"
}

#Initialize Selection to Null
model = None
test_path = "data/split/MBTIClassification_TestSet.csv" 

if model_option != "Select a Model":
    model_path = model_dict[model_option]
    model = load_model(model_path)

                        
# -----------------------------
# Upload Test Data
# -----------------------------
st.subheader("Test Data Source")
df=None
data_source = st.radio(
    "Select Test Data Option:",
    ("Select an option","Use Preloaded Test File", "Upload Your Own CSV")
)
if data_source == "Select an option":
    st.info("Please select a test data option to proceed.")
    st.stop()
elif data_source == "Use Preloaded Test File":
     
    try:
        df = pd.read_csv(test_path)
        st.success("Preloaded test file loaded successfully.")
        # Download button
        with open(test_path, "rb") as file:
            st.download_button(
                label="Download Test CSV",
                data=file,
                file_name="test.csv",
                mime="text/csv"
            )
    except FileNotFoundError:
        st.error("Preloaded test file not found in repository.")
        st.stop()

elif data_source == "Upload Your Own CSV":
    uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Please upload a test CSV file.")
        st.stop()



if df is not None and model is not None:
    is_valid, message = validate_test_file(df, model)

    if not is_valid:
        if st.button("Use Preloaded Test File Instead"):
            df = pd.read_csv(test_path)
            st.success("Switched to preloaded test file.")


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


    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("AUC", f"{auc:.4f}")
    col3.metric("Precision", f"{precision:.4f}")

    col4, col5, col6 = st.columns(3)

    col4.metric("Recall", f"{recall:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC", f"{mcc:.4f}")
 

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

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        output_dict=True
    )

    accuracy_value = report_dict["accuracy"]
    macro_f1 = report_dict["macro avg"]["f1-score"]
    weighted_f1 = report_dict["weighted avg"]["f1-score"]
    macro_precision = report_dict["macro avg"]["precision"]
    macro_recall = report_dict["macro avg"]["recall"]

    st.subheader("Overall Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy_value:.4f}")
    col2.metric("Macro F1 Score", f"{macro_f1:.4f}")
    col3.metric("Weighted F1 Score", f"{weighted_f1:.4f}")

    col4, col5 = st.columns(2)

    col4.metric("Macro Precision", f"{macro_precision:.4f}")
    col5.metric("Macro Recall", f"{macro_recall:.4f}")


    report_df = pd.DataFrame(report_dict).transpose()

    # Round values
    report_df = report_df.round(3)

    class_report = report_df.iloc[:-3]  # remove accuracy + averages
   
    st.subheader("Per-Class Performance")
    st.table(class_report)

 

