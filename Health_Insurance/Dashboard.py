# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import joblib

##### Load your dataset
path = ("/Users/stevenschepanski/Documents/Projects/Diabetes_Prediction/")
df_resized = pd.read_csv("data/df_resized.csv") 

# Data Overview Section
st.title("Diabetes Prediction Dashboard")

# Data Overview
st.sidebar.header("Data Overview")
st.sidebar.info("Displaying basic statistics of the dataset.")
st.sidebar.write("Shape of the Dataset:", df_resized.shape)
st.sidebar.write("Target Variable Distribution:")
st.sidebar.write(df_resized['outcome'].value_counts())

##### EDA Section
st.header("Exploratory Data Analysis")
st.write("Present pairplot for selected columns with the ability to choose outcome classes.")
# pairplot visualization
custom_colour = (108/255, 84/255, 158/255) # RGB(108, 84, 158)
selected_columns = ['Age', 'pregnancies', 'bmi', 'insulin', 'glucose', 'bloodpressure', 'diabetespedigreefunction',
                    'skinthickness', 'bloodpressure_2', 'outcome']
pairplot = sns.pairplot(df_resized[selected_columns], hue='outcome', palette={0: custom_colour, 1: 'orange'})
st.pyplot(pairplot)

##### Model Evaluation Section
st.header("Model Evaluation")

##### Logistic Regression
st.subheader("Logistic Regression")

# load model
random_search_log_regression = joblib.load(path + "models/random_search_log_regression.joblib")

# Extract features and target variable
X = df_resized.drop(columns=['Unnamed: 0', 'id', 'measurement_date', 'measurement_date_2'])
# Extract target variable
y = df_resized['outcome']
# Split the data into training and testing sets with stratification
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42, stratify=df_resized['outcome'])

# Confusion Matrix for Logistic Regression
st.subheader("Confusion Matrix for Logistic Regression")
conf_matrix_log_reg = confusion_matrix(ytest, random_search_log_regression.best_estimator_.predict(Xtest))
conf_matrix_fig = plt.figure()
sns.heatmap(conf_matrix_log_reg, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Logistic Regression')
st.pyplot(conf_matrix_fig)

# Calculate ROC Curve for Logistic Regression
st.subheader("ROC Curve for Logistic Regression")
fpr_log_reg, tpr_log_reg, _ = roc_curve(ytest, random_search_log_regression.best_estimator_.predict_proba(Xtest)[:, 1])
roc_fig = plt.figure()
plt.plot(fpr_log_reg, tpr_log_reg, color='darkorange', lw=2, label='Logistic Regression')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Logistic Regression')
plt.legend()
st.pyplot(roc_fig)

# Classification Report for Logistic Regression
st.subheader("Classification Report for Logistic Regression")
st.text(classification_report(ytest, random_search_log_regression.best_estimator_.predict(Xtest)))

##### Decision Tree
st.subheader("Decision Tree")

# load model
random_search_decision_tree = joblib.load(path + "models/random_search_decision_tree.joblib")

# Confusion Matrix for Decision Tree
st.subheader("Confusion Matrix for Decision Tree")
conf_matrix_dt = confusion_matrix(ytest, random_search_decision_tree.best_estimator_.predict(Xtest))
conf_matrix_dt_fig = plt.figure()
sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Decision Tree')
st.pyplot(conf_matrix_dt_fig)

# Calculate ROC Curve for Decision Tree
st.subheader("ROC Curve for Decision Tree")
fpr_dt, tpr_dt, _ = roc_curve(ytest, random_search_decision_tree.best_estimator_.predict_proba(Xtest)[:, 1])
roc_dt_fig = plt.figure()
plt.plot(fpr_dt, tpr_dt, color='darkorange', lw=2, label='Decision Tree')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Decision Tree')
plt.legend()
st.pyplot(roc_dt_fig)

# Classification Report for Decision Tree
st.subheader("Classification Report for Decision Tree")
st.text(classification_report(ytest, random_search_decision_tree.best_estimator_.predict(Xtest)))

##### Comparison of Accuracy Scores
st.subheader("Comparison of Accuracy Scores")
accuracy_scores = [random_search_log_regression.best_estimator_.score(Xtest, ytest), 
                   random_search_decision_tree.best_estimator_.score(Xtest, ytest)]
classifiers = ['Logistic Regression', 'Decision Tree']
accuracy_scores_fig = plt.figure()
sns.barplot(x=classifiers, y=accuracy_scores, palette='viridis')
plt.xlabel('Classifier Models')
plt.ylabel('Accuracy')
plt.title('Comparison of Logistic Regression and Decision Tree')
plt.ylim([0, 1])
st.pyplot(accuracy_scores_fig)

##### Conclusion Section
logistic_regression_summary = """
**Logistic Regression Performance Summary:**

- **Precision:** The model is 69% precise in identifying non-diabetic cases (0) and 57% precise in identifying diabetic cases (1). This indicates that when the model predicts a class, it is correct around 69% and 57% of the time for non-diabetic and diabetic cases, respectively.

- **Recall (Sensitivity):** The model exhibits high recall for non-diabetic cases (0) at 90%, suggesting it effectively captures true non-diabetic instances. However, the recall for diabetic cases (1) is lower at 25%, indicating a challenge in identifying true positive diabetic instances.

- **F1-Score:** The F1-score, a balance between precision and recall, is 0.78 for non-diabetic cases (0) and 0.35 for diabetic cases (1). The lower F1-score for diabetic cases reflects the trade-off between precision and recall in the model.

- **Accuracy:** The overall accuracy is 68%, indicating the proportion of correctly classified instances out of the total.
"""

decision_tree_summary = """
**Decision Tree Performance Summary:**

- **Precision:** The model achieves 78% precision in identifying non-diabetic cases (0) and 56% precision in identifying diabetic cases (1). This indicates that when the model predicts a class, it is correct around 78% and 56% of the time for non-diabetic and diabetic cases, respectively.

- **Recall (Sensitivity):** The model shows 75% recall for non-diabetic cases (0) and 60% recall for diabetic cases (1). This suggests that the decision tree effectively captures true non-diabetic instances but has a moderate ability to identify true positive diabetic instances.

- **F1-Score:** The F1-score, a balance between precision and recall, is 0.76 for non-diabetic cases (0) and 0.58 for diabetic cases (1). The F1-scores indicate a reasonable balance between precision and recall for both classes.

- **Accuracy:** The overall accuracy is 70%, indicating the proportion of correctly classified instances out of the total.
"""

# Display the performance summaries in the Streamlit app
st.header("Conclusion Section")
st.subheader("Logistic Regression Performance Summary")
st.markdown(logistic_regression_summary)

st.subheader("Decision Tree Performance Summary")
st.markdown(decision_tree_summary)
