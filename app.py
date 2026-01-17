import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from io import BytesIO

# ------------------------ Page config ------------------------
st.set_page_config(page_title="Confusion Matrix Generator", layout="centered")

# ------------------------ Centered Title ------------------------
st.markdown(
    "<h1 style='text-align: center;'>Confusion Matrix Generator</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Generate beautiful confusion matrices for your ML models</p>",
    unsafe_allow_html=True
)

st.info("You can either **upload a CSV file** containing true and predicted labels, or **enter the labels manually** in the fields below.")


st.markdown("---")

# ------------------------ CSV Upload ------------------------
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"],
    help="CSV must contain true and predicted labels (true/pred columns)."
)

# ------------------------ Manual input ------------------------
col1, col2 = st.columns(2)
with col1:
    true_labels_input = st.text_input(
        "True Labels (comma separated)",
        placeholder="Example: cat, dog, cat, cat, dog"
    )
with col2:
    pred_labels_input = st.text_input(
        "Predicted Labels (comma separated)",
        placeholder="Example: cat, cat, dog, cat, dog"
    )

# ------------------------ Color map selection ------------------------
cmap = st.selectbox(
    "Choose color map for the confusion matrix",
    options=[
        "viridis", "plasma", "inferno", "magma",
        "cividis", "Blues", "Greens", "coolwarm"
    ]
)

# ------------------------ Helper functions ------------------------
def parse_labels(text):
    """Parse manual input labels (numbers or strings)."""
    try:
        return [x.strip() for x in text.split(",")]
    except:
        return None

def detect_columns(df):
    true_cols = ["true", "t", "actual", "label"]
    pred_cols = ["pred", "predicted", "p"]

    true_col = next((c for c in df.columns if c.lower() in true_cols), None)
    pred_col = next((c for c in df.columns if c.lower() in pred_cols), None)

    return true_col, pred_col

# ------------------------ Custom button style ------------------------
st.markdown("""
<style>
.custom-button {
    background-color: #8CA9FF;
    color: black;
    font-weight: bold;
}
.custom-button:hover {
    background-color: #AAC4F5;
    color: black;
}
</style>
""", unsafe_allow_html=True)

# ------------------------ Generate Button ------------------------
generate_clicked = st.button("Generate Confusion Matrix", key="generate", help="Click to generate matrix", args=None)

# ------------------------ Load labels & generate ------------------------
if generate_clicked:
    true_labels = None
    pred_labels = None

    # ------------------------ CSV input ------------------------
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            true_col, pred_col = detect_columns(df)

            if true_col is None or pred_col is None:
                st.error("Could not detect true/pred columns in CSV.")
            else:
                true_labels = df[true_col].tolist()
                pred_labels = df[pred_col].tolist()
                st.success("CSV loaded successfully ✅")
        except:
            st.error("Error reading CSV file.")

    # ------------------------ Manual input fallback ------------------------
    if true_labels_input and pred_labels_input:
        true_labels = parse_labels(true_labels_input)
        pred_labels = parse_labels(pred_labels_input)

        if true_labels is None or pred_labels is None:
            st.error("Invalid manual input. Use numbers or strings separated by commas.")

    # ------------------------ Validation ------------------------
    if true_labels is not None and pred_labels is not None:
        if len(true_labels) != len(pred_labels):
            st.error("True labels and predicted labels must have the same length.")
        else:
            # ------------------------ Confusion Matrix ------------------------
            classes = sorted(list(set(true_labels) | set(pred_labels)))  # works for strings and numbers
            cm = confusion_matrix(true_labels, pred_labels, labels=classes)

            fig, ax = plt.subplots(figsize=(6,5))
            im = ax.imshow(cm, cmap=cmap)

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j, i, cm[i, j],
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > cm.max()/2 else "black"
                    )

            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            ax.set_xticks(range(len(classes)))
            ax.set_yticks(range(len(classes)))
            ax.set_xticklabels(classes)
            ax.set_yticklabels(classes)
            ax.set_title("Confusion Matrix")
            plt.colorbar(im, ax=ax)

            st.pyplot(fig)

            # ------------------------ Metrics ------------------------
            metrics_df = pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
                "Value": [
                    f"{accuracy_score(true_labels, pred_labels):.4f}",
                    f"{precision_score(true_labels, pred_labels, average='weighted', zero_division=0):.4f}",
                    f"{recall_score(true_labels, pred_labels, average='weighted', zero_division=0):.4f}",
                    f"{f1_score(true_labels, pred_labels, average='weighted', zero_division=0):.4f}"
                ]
            })

            st.subheader("Metrics")
            st.table(metrics_df)

            # ------------------------ Download Button ------------------------
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)

            st.download_button(
                label="📥 Download Confusion Matrix",
                data=buf,
                file_name="confusion_matrix.png",
                mime="image/png"
            )

# ------------------------ Footer ------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Made with ❤️ by Paresh Nayak</p>",
    unsafe_allow_html=True
)




