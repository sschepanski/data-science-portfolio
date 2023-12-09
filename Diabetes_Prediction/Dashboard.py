# app.py
import streamlit as st

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix, classification_report

# Load your dataset
df_resized = pd.read_csv("/Users/stevenschepanski/Documents/Projects/Diabetes_Prediction/data/df_resized.csv") 

# Data Overview Section
st.title("Diabetes Prediction Dashboard")

# Data Overview
st.sidebar.header("Data Overview")
st.sidebar.info("Displaying basic statistics of the dataset.")
st.sidebar.write("Shape of the Dataset:", df_resized.shape)
st.sidebar.write("Target Variable Distribution:")
st.sidebar.write(df_resized['outcome'].value_counts())

# EDA Section
st.header("Exploratory Data Analysis")
st.write("Present pairplot for selected columns with the ability to choose outcome classes.")
# pairplot visualization
custom_colour = (108/255, 84/255, 158/255) # RGB(108, 84, 158)
selected_columns = ['Age', 'pregnancies', 'bmi', 'insulin', 'glucose', 'bloodpressure', 'diabetespedigreefunction',
                    'skinthickness', 'bloodpressure_2', 'outcome']
sns.pairplot(df_resized[selected_columns], hue='outcome', palette={0: custom_colour, 1: 'orange'})

# Show the pairplot
plt.show()

# Model Evaluation Section
st.header("Model Evaluation")

## Logistic Regression
st.subheader("Logistic Regression")

# Plot ROC Curve
st.subheader("ROC Curve for Logistic Regression")
fpr, tpr, _ = roc_curve(ytest, random_search.best_estimator_.predict_proba(Xtest)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Logistic Regression')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Logistic Regression')
plt.legend()
st.pyplot(plt)

# Confusion Matrix
st.subheader("Confusion Matrix for Logistic Regression")
conf_matrix = confusion_matrix(ytest, random_search.best_estimator_.predict(Xtest))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Logistic Regression')
st.pyplot(plt)

# Classification Report
st.subheader("Classification Report for Logistic Regression")
st.text(classification_report(ytest, random_search.best_estimator_.predict(Xtest)))

# Decision Tree
st.subheader("Decision Tree")

# Plot ROC Curve
st.subheader("ROC Curve for Decision Tree")
fpr_dt, tpr_dt, _ = roc_curve(ytest, random_search_decision_tree.best_estimator_.predict_proba(Xtest)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr_dt, tpr_dt, color='darkorange', lw=2, label='Decision Tree')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Decision Tree')
plt.legend()
st.pyplot(plt)

# Confusion Matrix
st.subheader("Confusion Matrix for Decision Tree")
conf_matrix_dt = confusion_matrix(ytest, random_search_decision_tree.best_estimator_.predict(Xtest))
sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Decision Tree')
st.pyplot(plt)

# Classification Report
st.subheader("Classification Report for Decision Tree")
st.text(classification_report(ytest, random_search_decision_tree.best_estimator_.predict(Xtest)))

# Comparison of Accuracy Scores
st.subheader("Comparison of Accuracy Scores")
accuracy_scores = [random_search.best_estimator_.score(Xtest, ytest), 
                   random_search_decision_tree.best_estimator_.score(Xtest, ytest)]
classifiers = ['Logistic Regression', 'Decision Tree']
plt.figure(figsize=(8, 6))
sns.barplot(x=classifiers, y=accuracy_scores, palette='viridis')
plt.xlabel('Classifier Models')
plt.ylabel('Accuracy')
plt.title('Comparison of Logistic Regression and Decision Tree')
plt.ylim([0, 1])
st.pyplot(plt)

# Conclusion Section
st.header("Conclusion")

st.subheader("Logistic Regression Performance Summary")
st.text("The logistic regression model shows mixed performance in predicting diabetes:")
# ... (Add your performance summary for logistic regression)

st.subheader("Decision Tree Performance Summary")
st.text("The decision tree model exhibits the following performance metrics:")
# ... (Add your performance summary for decision tree)

# Save the app.py file and run the Streamlit app using "streamlit run app.py" in the terminal.
