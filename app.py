import streamlit as st
import pandas as pd
import os
import re
import string
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Spam Detection Platform", page_icon="📩", layout="wide")


# ---------------- PATHS ----------------
RAW_DIR = "data/raw"
CLEANED_DIR = "data/cleaned"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEANED_DIR, exist_ok=True)


# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = str(text)
    if text.lower() == "nan":
        text = ""
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------- READ CSV SAFE ----------------
def read_spam_csv(file):
    df = pd.read_csv(file, encoding="latin1")

    # common spam dataset: v1(label), v2(message)
    if "v1" in df.columns and "v2" in df.columns:
        df = df[["v1", "v2"]]
        df.columns = ["label", "message"]

    # already clean format
    elif "label" in df.columns and "message" in df.columns:
        df = df[["label", "message"]]

    else:
        # fallback: first 2 columns
        df = df.iloc[:, :2]
        df.columns = ["label", "message"]

    return df


# ---------------- CLEAN DATASET ----------------
def clean_dataset(df, strategy="drop"):
    df = df.copy()

    # force string conversion
    df["label"] = df["label"].astype(str)
    df["message"] = df["message"].astype(str)

    # handle missing values
    if strategy == "drop":
        df = df.dropna(subset=["label", "message"])
    elif strategy == "fill":
        df["label"] = df["label"].fillna("ham")
        df["message"] = df["message"].fillna("")

    # remove invalid text rows
    df["message"] = df["message"].replace("nan", "")
    df = df[df["message"].str.strip() != ""]

    # normalize labels
    df["label"] = df["label"].str.lower().str.strip()
    df["label"] = df["label"].replace({"0": "ham", "1": "spam"})

    # keep only spam/ham
    df = df[df["label"].isin(["spam", "ham"])]

    # clean messages
    df["message"] = df["message"].apply(clean_text)

    # remove empty after cleaning
    df = df[df["message"].str.strip() != ""]

    # label binary
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

    return df


# ---------------- TRAIN MODEL ----------------
@st.cache_resource
def train_model(df):
    df = df.copy()

    # final safety for NaN
    df["message"] = df["message"].fillna("").astype(str)
    df = df[df["message"].str.strip() != ""]

    X = df["message"]
    y = df["label_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return model, vectorizer, acc, report, cm


# ---------------- UI ----------------
st.title("📩 End to End Spam Detection Platform (Naive Bayes)")
st.write("---")


# =========================================================
# STEP 1: DATA INGESTION
# =========================================================
st.subheader("Step 1: Data Ingestion")

source = st.radio(
    "Choose Data Source",
    ["Use Default Dataset (data/raw/spam.csv)", "Upload CSV"],
    horizontal=True
)

df_raw = None

if source == "Use Default Dataset (data/raw/spam.csv)":
    default_path = os.path.join(RAW_DIR, "spam.csv")

    if os.path.exists(default_path):
        df_raw = read_spam_csv(default_path)
        st.success("✅ Default dataset loaded successfully!")
    else:
        st.error("❌ Default dataset not found at: data/raw/spam.csv")
        st.stop()

else:
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        df_raw = read_spam_csv(uploaded_file)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(RAW_DIR, f"uploaded_spam_{timestamp}.csv")
        df_raw.to_csv(save_path, index=False)

        st.success("✅ File uploaded successfully!")
        st.info(f"Saved as: {save_path}")

if df_raw is None:
    st.warning("⚠️ Please upload a dataset or use default dataset.")
    st.stop()

st.write("### Dataset Preview")
st.dataframe(df_raw.head(10), use_container_width=True)
st.write(f"**Shape:** {df_raw.shape}")

st.write("### Missing Values")
st.dataframe(df_raw.isnull().sum().to_frame("Missing Count"), use_container_width=True)

st.write("---")


# =========================================================
# STEP 2: DATA CLEANING
# =========================================================
st.subheader("Step 2: Data Cleaning")

clean_strategy = st.selectbox(
    "Missing Values Strategy",
    ["drop", "fill"],
    format_func=lambda x: "Drop Null Rows" if x == "drop" else "Fill Missing Values"
)

if st.button("Run Data Cleaning"):
    df_cleaned = clean_dataset(df_raw, strategy=clean_strategy)
    st.session_state["cleaned_df"] = df_cleaned
    st.success("✅ Data Cleaning Completed!")

if "cleaned_df" in st.session_state:
    st.write("### Cleaned Dataset Preview")
    st.dataframe(st.session_state["cleaned_df"].head(10), use_container_width=True)
    st.write(f"**Shape (Cleaned):** {st.session_state['cleaned_df'].shape}")

    if st.button("Save Cleaned Dataset"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cleaned_name = f"cleaned_spam_{ts}.csv"
        cleaned_path = os.path.join(CLEANED_DIR, cleaned_name)
        st.session_state["cleaned_df"].to_csv(cleaned_path, index=False)
        st.success(f"✅ Cleaned dataset saved as: {cleaned_name}")

st.write("---")


# =========================================================
# STEP 3: LOAD CLEANED DATASET
# =========================================================
st.subheader("Step 3: Load Cleaned Dataset")

cleaned_files = [f for f in os.listdir(CLEANED_DIR) if f.endswith(".csv")]

if len(cleaned_files) == 0:
    st.warning("⚠️ No cleaned datasets found. Please clean and save first.")
    st.stop()

selected_cleaned = st.selectbox("Select Cleaned Dataset", cleaned_files)

cleaned_path = os.path.join(CLEANED_DIR, selected_cleaned)
df_final = pd.read_csv(cleaned_path)

st.success(f"✅ Loaded cleaned dataset: {selected_cleaned}")
st.dataframe(df_final.head(10), use_container_width=True)

st.write("---")


# =========================================================
# STEP 4: MODEL TRAINING
# =========================================================
st.subheader("Step 4: Model Training (Naive Bayes)")

# if label_num missing, create it
if "label_num" not in df_final.columns:
    df_final["label"] = df_final["label"].astype(str).str.lower().str.strip()
    df_final["label"] = df_final["label"].replace({"0": "ham", "1": "spam"})
    df_final = df_final[df_final["label"].isin(["spam", "ham"])]
    df_final["label_num"] = df_final["label"].map({"ham": 0, "spam": 1})

# final safety for NaN messages
df_final["message"] = df_final["message"].fillna("").astype(str)
df_final = df_final[df_final["message"].str.strip() != ""]

model, vectorizer, acc, report, cm = train_model(df_final)

st.info(f"✅ Model Trained Successfully | Accuracy: **{acc*100:.2f}%**")

with st.expander("📊 Classification Report"):
    st.text(report)

with st.expander("🧾 Confusion Matrix"):
    st.write(cm)

st.write("---")


# =========================================================
# STEP 5: PREDICTION
# =========================================================
st.subheader("Step 5: Spam Prediction")

user_msg = st.text_area("Enter a message to check Spam / Ham:")

if st.button("Predict Message"):
    if user_msg.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        msg_clean = clean_text(user_msg)
        msg_vect = vectorizer.transform([msg_clean])
        pred = model.predict(msg_vect)[0]

        if pred == 1:
            st.error("🚨 Spam Message Detected!")
        else:
            st.success("✅ Not Spam (Ham) Message")
