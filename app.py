# ============================================================
# Streamlit App - HND Specialization Recommender & Analytics
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

# ------------------------------
# 1. Load Dataset, Model & Preprocessor
# ------------------------------
df = pd.read_csv('synthetic_hnd_dataset_2000.csv')
xgb_model = joblib.load('xgb_hnd_model.pkl')
preprocessor = joblib.load('xgb_preprocessor_hnd.pkl')

# HND Classes
hnd_classes = [
    'HND in Artificial Intelligence',
    'HND in Cyber Security and Data Protection',
    'HND in Networking and Cloud Computing',
    'HND in Software and Web Development'
]

# ND courses used as features
nd_courses = [
    'CSC 111','CSC 112','CSC 113','CSC 121','CSC 122','CSC 123',
    'CSC 211','CSC 212','CSC 213','CSC 221','CSC 222','CSC 223',
    'MTH 111','MTH 112','MTH 121','MTH 122','MTH 211','MTH 212','MTH 221','MTH 222'
]

career_goal_options = df['career_goal'].unique().tolist()

# ------------------------------
# 2. Sidebar Navigation
# ------------------------------
st.set_page_config(page_title="HND Recommender & Analytics", layout="wide")
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose Page", 
                                ["Recommender", "Confusion Matrix", "ROC Curve", "Classification Report", "Sample Text"])

# ------------------------------
# 3. Recommender Page
# ------------------------------
if app_mode == "Recommender":
    st.title("📘 HND Specialization Recommender")
    st.write("Generate a random ND profile to get HND recommendation based on ND course scores only.")

    st.sidebar.header("ND Grades Input")
    user_input = {}

    # Generate random ND scores from dataset
    if st.sidebar.button("Generate Random ND Profile"):
        random_row = df.sample(1).iloc[0]
        for course in nd_courses:
            # Random variation around the dataset value
            user_input[course] = int(np.clip(random_row[course] + np.random.randint(-10, 10), 40, 100))
        # Add dummy career_goal for preprocessor
        user_input['career_goal'] = random.choice(career_goal_options)
        st.sidebar.write("Random ND profile generated!")

    else:
        # Manual input
        for course in nd_courses:
            user_input[course] = st.sidebar.number_input(f"{course} Score", min_value=0, max_value=100, value=random.randint(50, 100))
        # Dummy career_goal
        user_input['career_goal'] = random.choice(career_goal_options)

    # Convert input to DataFrame
    input_df = pd.DataFrame([user_input])

    if st.button("Get Recommendation"):
        try:
            # Transform input with preprocessor
            X_processed = preprocessor.transform(input_df)
            y_pred_prob = xgb_model.predict_proba(X_processed)
            y_pred_class = np.argmax(y_pred_prob, axis=1)[0]
            recommended_hnd = hnd_classes[y_pred_class]

            st.success(f"Recommended HND Specialization: **{recommended_hnd}**")

            st.subheader("Prediction Probabilities")
            prob_df = pd.DataFrame(y_pred_prob, columns=hnd_classes)
            st.dataframe(prob_df.T.rename(columns={0: "Probability"}))

            st.subheader("Input Profile Used")
            st.dataframe(input_df.T.rename(columns={0: "Value"}))

        except Exception as e:
            st.error(f"Error: {e}")

# ------------------------------
# 4. Confusion Matrix
# ------------------------------
elif app_mode == "Confusion Matrix":
    st.title("Confusion Matrix")
    st.write("Confusion matrix of the trained XGBoost model.")

    X = preprocessor.transform(df[nd_courses + ['career_goal']])
    y_true = df['hnd_encoded'].values
    y_pred = xgb_model.predict(X)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=hnd_classes, yticklabels=hnd_classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# ------------------------------
# 5. ROC Curve
# ------------------------------
elif app_mode == "ROC Curve":
    st.title("ROC Curve")
    st.write("ROC curve for each HND class.")

    X = preprocessor.transform(df[nd_courses + ['career_goal']])
    y_true = df['hnd_encoded'].values
    y_prob = xgb_model.predict_proba(X)

    fig, ax = plt.subplots()
    for i, class_name in enumerate(hnd_classes):
        fpr, tpr, _ = roc_curve((y_true==i).astype(int), y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve by Class")
    ax.legend(loc="lower right")
    st.pyplot(fig)

# ------------------------------
# 6. Classification Report
# ------------------------------
elif app_mode == "Classification Report":
    st.title("Classification Report")
    X = preprocessor.transform(df[nd_courses + ['career_goal']])
    y_true = df['hnd_encoded'].values
    y_pred = xgb_model.predict(X)
    report = classification_report(y_true, y_pred, target_names=hnd_classes, output_dict=True)
    report_df = pd.DataFrame(report).T
    st.dataframe(report_df)

# ------------------------------
# 7. Sample Text
# ------------------------------
elif app_mode == "Sample Text":
    st.title("Sample Text Generator (Sentiment)")
    st.write("Generate random sample sentences with positive or negative sentiment.")

    positive_samples = [
        "I love studying computer science!",
        "The AI course is fascinating and well structured.",
        "I am confident I will excel in networking and cloud computing."
    ]
    negative_samples = [
        "I struggle with programming assignments.",
        "Database management is confusing and stressful.",
        "I am unsure about my future career path."
    ]

    if st.button("Generate Random Positive Text"):
        st.success(random.choice(positive_samples))

    if st.button("Generate Random Negative Text"):
        st.error(random.choice(negative_samples))
