# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import joblib

# Load your dataset and models
path = ("/Users/stevenschepanski/Documents/Projects/Diabetes_Prediction/")
df_resized = pd.read_csv("data/df_resized.csv") 
random_search_log_regression = joblib.load(path + "models/random_search_log_regression.joblib")
random_search_decision_tree = joblib.load(path + "models/random_search_decision_tree.joblib")

# Set page width to the maximum
st.set_page_config(layout="wide")

# Data Overview Section
st.title("Diabetes Prediction Dashboard")

# Display general statistics on the top of the page
total_samples = df_resized.shape[0]
percentage_no_diabetes = (df_resized['outcome'] == 0).sum() / total_samples * 100
percentage_diabetes = (df_resized['outcome'] == 1).sum() / total_samples * 100

# Arrange the general stats in three columns
columns = st.columns(3)

# For Total Sample Size
columns[0].markdown("<div style='text-align: center;'><b>Total Sample Size</b></div>", unsafe_allow_html=True)
columns[0].markdown(f"<div style='text-align: center; font-size: 36px;'>{total_samples}</div>", unsafe_allow_html=True)

# For Percentage of No Diabetes
columns[1].markdown("<div style='text-align: center;'><b>Percentage of No Diabetes</b></div>", unsafe_allow_html=True)
columns[1].markdown(f"<div style='text-align: center; font-size: 36px;'>{percentage_no_diabetes:.2f}%</div>", unsafe_allow_html=True)

# For Percentage of Diabetes
columns[2].markdown("<div style='text-align: center;'><b>Percentage of Diabetes</b></div>", unsafe_allow_html=True)
columns[2].markdown(f"<div style='text-align: center; font-size: 36px;'>{percentage_diabetes:.2f}%</div>", unsafe_allow_html=True)

# Add a separator
st.markdown("<hr style='margin: 20px;'>", unsafe_allow_html=True)

# Sidebar for Data Overview and EDA
st.sidebar.header("Data Overview")

##### EDA Section
# Sidebar for EDA options
selected_columns = st.sidebar.multiselect(
    "Select 2 Features for EDA",
    [
        "pregnancies",
        "insulin",
        "glucose",
        "bloodpressure",
        "diabetespedigreefunction",
        "skinthickness",
        "bloodpressure_2",
    ],
)

# Slider for Age and BMI filtering
age_range = st.sidebar.slider(
    "Select Age Range",
    min_value=df_resized["Age"].min(),
    max_value=df_resized["Age"].max(),
    value=(df_resized["Age"].min(), df_resized["Age"].max()),
)
bmi_range = st.sidebar.slider(
    "Select BMI Range",
    min_value=df_resized["bmi"].min(),
    max_value=df_resized["bmi"].max(),
    value=(df_resized["bmi"].min(), df_resized["bmi"].max()),
)

# Filter data based on selected ranges
filtered_data = df_resized[
    (df_resized["Age"] >= age_range[0])
    & (df_resized["Age"] <= age_range[1])
    & (df_resized["bmi"] >= bmi_range[0])
    & (df_resized["bmi"] <= bmi_range[1])
]

# Select fixed hue for the scatter plot
fixed_hue = st.sidebar.selectbox("Fixed Outcome Classes", ["outcome"])

# Choose the type of plot
plot_type = st.sidebar.selectbox("Select Plot Type", ["scatter"])

# Scatter plot based on user selection
custom_colour = (108 / 255, 84 / 255, 158 / 255)  # RGB(108, 84, 158)
if plot_type == "scatter" and len(selected_columns) == 2:
    # Show selected plot_type in the second line as the first column
    st.subheader(f"Selected Plot Type: {plot_type}")

    # Create a Matplotlib Figure for the scatter plot with larger size
    fig, (ax_scatter, ax_hist1, ax_hist2) = plt.subplots(
        1, 3, figsize=(25, 5)
    )

    # Plot the scatter plot using Matplotlib
    scatter_plot = sns.scatterplot(
        data=df_resized,
        x=selected_columns[0],
        y=selected_columns[1],
        hue=fixed_hue,
        palette={0: custom_colour, 1: "orange"},
        ax=ax_scatter,
    )

    # Get histogram bin size from user selection
    bin_size = st.slider(
        "Select Bin Size for Histograms", min_value=1, max_value=100, value=30
    )

    # Plot the histograms for each selected feature with the chosen bin size
    for i, feature in enumerate(selected_columns):
        # Include outcome differences in histograms
        sns.histplot(
            data=df_resized,
            x=feature,
            hue="outcome",
            multiple="stack",
            bins=bin_size,
            palette={0: custom_colour, 1: "orange"},
            ax=ax_hist1 if i == 0 else ax_hist2,
        )
        ax_hist1.set_title(f"Histogram of {selected_columns[0]}")
        ax_hist2.set_title(f"Histogram of {selected_columns[1]}")

    # Set titles for the plots
    ax_scatter.set_title(
        f"Scatter Plot of {selected_columns[0]} and {selected_columns[1]}"
    )
    ax_hist1.set_title(f"Histogram of {selected_columns[0]}")
    ax_hist2.set_title(f"Histogram of {selected_columns[1]}")

    # Show the scatter plot and histograms using Streamlit
    st.pyplot(fig)


# Sidebar for Model Evaluation options
selected_model = st.sidebar.selectbox("Select Model", ["", "Logistic Regression", "Decision Tree"])

# Extract features and target variable
X = df_resized.drop(columns=['Unnamed: 0', 'id', 'measurement_date', 'measurement_date_2'])
# Extract target variable
y = df_resized['outcome']
# Split the data into training and testing sets with stratification
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42, stratify=df_resized['outcome'])

# Logistic Regression
if selected_model:
    if selected_model == "Logistic Regression":
        ##### Model Evaluation Section
        st.subheader("Model Evaluation: Logistic Regression")

        # Confusion Matrix and ROC Curve in one row
        columns = st.columns(2)

        # Confusion Matrix for Logistic Regression
        with columns[0]:
            st.subheader("Confusion Matrix for Logistic Regression")
            conf_matrix_log_reg = confusion_matrix(ytest, random_search_log_regression.best_estimator_.predict(Xtest))
            conf_matrix_fig = plt.figure()
            sns.heatmap(conf_matrix_log_reg, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(conf_matrix_fig)

        # ROC Curve for Logistic Regression
        with columns[1]:
            st.subheader("ROC Curve for Logistic Regression")
            fpr_log_reg, tpr_log_reg, _ = roc_curve(ytest, random_search_log_regression.best_estimator_.predict_proba(Xtest)[:, 1])
            roc_fig = plt.figure()
            plt.plot(fpr_log_reg, tpr_log_reg, color='darkorange', lw=2, label='Logistic Regression')
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            st.pyplot(roc_fig)

        # Classification Report for Logistic Regression
        st.subheader("Classification Report for Logistic Regression")
        st.text(classification_report(ytest, random_search_log_regression.best_estimator_.predict(Xtest)))

    # Decision Tree
    elif selected_model == "Decision Tree":
        st.subheader("Model Evaluation: Decision Tree")

        # Confusion Matrix and ROC Curve in one row
        columns = st.columns(2)

        # Confusion Matrix for Decision Tree
        with columns[0]:
            st.subheader("Confusion Matrix for Decision Tree")
            conf_matrix_dt = confusion_matrix(ytest, random_search_decision_tree.best_estimator_.predict(Xtest))
            conf_matrix_dt_fig = plt.figure()
            sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(conf_matrix_dt_fig)

        # ROC Curve for Decision Tree
        with columns[1]:
            st.subheader("ROC Curve for Decision Tree")
            fpr_dt, tpr_dt, _ = roc_curve(ytest, random_search_decision_tree.best_estimator_.predict_proba(Xtest)[:, 1])
            roc_dt_fig = plt.figure()
            plt.plot(fpr_dt, tpr_dt, color='darkorange', lw=2, label='Decision Tree')
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            st.pyplot(roc_dt_fig)

        # Classification Report for Decision Tree
        st.subheader("Classification Report for Decision Tree")
        st.text(classification_report(ytest, random_search_decision_tree.best_estimator_.predict(Xtest)))

    else:
        st.warning("Please select a model.")

# Comparison of Accuracy Scores
selected_models = st.sidebar.multiselect(
    "Select Models for Comparison", ["Logistic Regression", "Decision Tree"]
)

# Display accuracy scores on the main page only when models are selected
if selected_models:
    st.subheader("Comparison of Accuracy Scores")

    # Create a DataFrame for plotting
    data = {"Model": [], "Accuracy": []}

    for model in selected_models:
        accuracy_score = None
        if model == "Logistic Regression":
            accuracy_score = random_search_log_regression.best_estimator_.score(Xtest, ytest)
        elif model == "Decision Tree":
            accuracy_score = random_search_decision_tree.best_estimator_.score(Xtest, ytest)

        if accuracy_score is not None:
            data["Model"].append(model)
            data["Accuracy"].append(accuracy_score)

    df_accuracy = pd.DataFrame(data)

    # Plot comparison of accuracy scores on the main page with custom colors
    accuracy_scores_fig = plt.figure()
    palette = {"Logistic Regression": custom_colour, "Decision Tree": "orange"}
    sns.barplot(x="Model", y="Accuracy", data=df_accuracy, palette=palette)
    plt.ylabel("Accuracy")
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

# Conclusion Section
st.sidebar.header("Conclusion Section")

# Sidebar for Conclusion options
conclusion_options = st.sidebar.multiselect("Select Models for Conclusion", ["Logistic Regression", "Decision Tree"])

# Display conclusion in the sidebar
if conclusion_options:
    for model in conclusion_options:
        if model == "Logistic Regression":
            st.sidebar.subheader("Logistic Regression Performance Summary")
            st.sidebar.markdown(logistic_regression_summary)
        elif model == "Decision Tree":
            st.sidebar.subheader("Decision Tree Performance Summary")
            st.sidebar.markdown(decision_tree_summary)
        else:
            st.sidebar.warning(f"No conclusion available for {model}.")

