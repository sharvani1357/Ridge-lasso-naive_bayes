# 📩 End to End Spam Detection System (Naive Bayes + Streamlit)

A complete **Streamlit web application** that predicts whether a given SMS/Email message is **Spam** or **Not Spam (Ham)** using **Naive Bayes** classification with **TF-IDF** text vectorization.

---

## 📌 Project Features

✅ **Spam Detection (Real-Time)**  
Users can enter any message and instantly check whether it is Spam or Ham.

✅ **Dataset Upload Support**  
- Use default dataset from `data/raw/spam.csv`  
- Or upload your own CSV file directly from the app

✅ **Exploratory Data Analysis (EDA)**  
Displays:
- Dataset preview  
- Shape of dataset  
- Missing values count  

✅ **Data Cleaning Module**  
Supports cleaning strategies:
- Drop null values  
- Fill missing values  

✅ **Save & Load Cleaned Dataset**  
- Saves cleaned datasets into `data/cleaned/`  
- Allows selecting any cleaned file for training

✅ **Model Training + Evaluation**  
Shows:
- Accuracy score  
- Classification report  
- Confusion matrix  

---

## 🧠 ML Workflow Used

1. **Data Loading**
   - Reads dataset from:
     - `data/raw/spam.csv` *(default)*  
     - or uploaded CSV file

2. **Data Cleaning**
   - Handles missing values using:
     - Drop missing rows OR Fill missing values  
   - Removes empty/invalid messages  
   - Standardizes labels (`spam`, `ham`)

3. **Text Preprocessing**
   - Converts text to lowercase  
   - Removes:
     - punctuation  
     - numbers  
     - extra spaces  

4. **Feature Extraction**
   - Uses **TF-IDF Vectorizer**
   - Converts text into numeric form for ML model

5. **Model Training**
   - Uses **Multinomial Naive Bayes**
   - Trains on cleaned dataset

6. **Model Evaluation**
   - Displays:
     - Accuracy  
     - Classification report  
     - Confusion matrix  

7. **Prediction**
   - User enters a message  
   - App predicts:
     - 🚨 Spam  
     - ✅ Ham  

---

## 📁 Project Structure

### Spam-Detection/
### ├── app.py  
### ├── requirements.txt  
### ├── README.md  
### └── data/  
###  ├── raw/  
###  │   └── spam.csv  
###  └── cleaned/  
###      └── cleaned_spam_YYYYMMDD_HHMMSS.csv  

---

## ⚙️ Installation

### 1) Install Dependencies

 - pip install -r requirements.txt

## ▶️ Run the Streamlit App
### If streamlit command is not working, run using:
 - python -m streamlit run app.py

## 📌 Input Fields (User Inputs)

### The application allows the user to:

 - Upload dataset CSV (optional)
 - Select missing value cleaning strategy
 - Save cleaned dataset
 - Load cleaned dataset
 - Enter a message to check spam/ham

## 🔘 Prediction Button

### Click the button below to generate the result:

 - ✅ Predict Message
 - 📊 Output (Prediction Result)

### The application displays the result clearly as:

 - 🚨 Spam Message Detected! (Red Highlight)
 - ✅ Not Spam (Ham) Message (Green Highlight)

### It also shows:

 - Model Accuracy
 - Classification Report
 - Confusion Matrix

## 💡 Business Explanation

### This project helps in:

 - Automatically filtering spam SMS / emails
 - Reducing fraud messages and unwanted promotions
 - Improving user experience by keeping inbox clean

### Example reasoning:

 - “This message contains promotional keywords and patterns commonly found in spam.”
 - “This message looks like a normal personal message, so it is classified as Ham.”
