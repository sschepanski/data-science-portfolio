# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import joblib

# Load your dataset and models
path = ("/Users/stevenschepanski/Documents/Projects/Health_Insurance/")
df = pd.read_csv("data/insurance_modified.csv") 
random_search_decision_tree = joblib.load(path + "models/random_search_decision_tree.joblib")
random_search_lasso_regression = joblib.load(path + "models/random_search_lasso_regression.joblib")
random_search_lightgbm = joblib.load(path + "models/random_search_lightgbm.joblib")
random_search_random_forest = joblib.load(path + "models/random_search_random_forest.joblib")
random_search_ridge_regression = joblib.load(path + "models/random_search_ridge_regression.joblib")
random_search_svm = joblib.load(path + "models/random_search_svm.joblib")
random_search_xgboost = joblib.load(path + "models/random_search_xgboost.joblib")

# Set page width to the maximum
st.set_page_config(layout="wide")

# Data Overview Section
st.title("Health Insurance Cost Dashboard")

# Display general statistics on the top of the page
# Calculate average charges, age, and BMI for all, males, and females
average_charges_all = df['charges'].mean()
average_age_all = df['age'].mean()
average_bmi_all = df['bmi'].mean()

average_charges_male = df[df['sex'] == 1]['charges'].mean()
average_age_male = df[df['sex'] == 1]['age'].mean()
average_bmi_male = df[df['sex'] == 1]['bmi'].mean()

average_charges_female = df[df['sex'] == 0]['charges'].mean()
average_age_female = df[df['sex'] == 0]['age'].mean()
average_bmi_female = df[df['sex'] == 0]['bmi'].mean()

# Arrange the general stats in three columns
columns = st.columns(3)

# Average charges
columns[0].markdown("<div style='text-align: center;'><b>Average Charges</b></div>", unsafe_allow_html=True)
columns[0].markdown(f"<div style='text-align: center; font-size: 24px;'>{average_charges_all:.2f}</div>", unsafe_allow_html=True)
columns[1].markdown(f"<div style='text-align: center; font-size: 24px;'>{average_charges_male:.2f} ♂</div>", unsafe_allow_html=True)
columns[2].markdown(f"<div style='text-align: center; font-size: 24px;'>{average_charges_female:.2f} ♀</div>", unsafe_allow_html=True)

# Average age
columns[0].markdown("<div style='text-align: center;'><b>Average Age</b></div>", unsafe_allow_html=True)
columns[0].markdown(f"<div style='text-align: center; font-size: 24px;'>{average_age_all:.2f}</div>", unsafe_allow_html=True)
columns[1].markdown(f"<div style='text-align: center; font-size: 24px;'>{average_age_male:.2f} ♂</div>", unsafe_allow_html=True)
columns[2].markdown(f"<div style='text-align: center; font-size: 24px;'>{average_age_female:.2f} ♀</div>", unsafe_allow_html=True)

# Average BMI
columns[0].markdown("<div style='text-align: center;'><b>Average BMI</b></div>", unsafe_allow_html=True)
columns[0].markdown(f"<div style='text-align: center; font-size: 24px;'>{average_bmi_all:.2f}</div>", unsafe_allow_html=True)
columns[1].markdown(f"<div style='text-align: center; font-size: 24px;'>{average_bmi_male:.2f} ♂</div>", unsafe_allow_html=True)
columns[2].markdown(f"<div style='text-align: center; font-size: 24px;'>{average_bmi_female:.2f} ♀</div>", unsafe_allow_html=True)

# Add a separator
st.markdown("<hr style='margin: 20px;'>", unsafe_allow_html=True)

# Sidebar for Data Overview and EDA
st.sidebar.header("Data Overview")

##### EDA Section
# Sidebar for EDA options
selected_columns = st.sidebar.multiselect(
    "Select 2 Features for EDA",
    ["charges", "children", "bmi", "age"],
)

# Create two dots to select either the two sexes, the two smoker sections, the 4 regions, and 4 bmi_classes
selected_sex = st.sidebar.multiselect("Select Sex", df["sex"].unique())
selected_smoker = st.sidebar.multiselect("Select Smoker", df["smoker"].unique())
selected_region = st.sidebar.multiselect("Select Region", df["region"].unique())
selected_bmi_class = st.sidebar.multiselect("Select BMI Class", df["bmi_class"].unique())

# Choose the type of plot
plot_type = st.sidebar.selectbox("Select Plot Type", ["scatter"])

# Choose color
custom_colour = (108 / 255, 84 / 255, 158 / 255)  # RGB(108, 84, 158)

# Scatter plot based on user selection
if plot_type == "scatter" and len(selected_columns) == 2:
    # Show selected plot_type in the second line as the first column
    st.subheader(f"Selected Plot Type: {plot_type}")

    # Create a Matplotlib Figure for the scatter plot with larger size
    fig, (ax_scatter, ax_hist1, ax_hist2) = plt.subplots(1, 3, figsize=(25, 5))

    # Filter data based on selected options, but only if options are selected
    if selected_sex or selected_smoker or selected_region or selected_bmi_class:
        filtered_data = df[
            (df["sex"].isin(selected_sex))
            | (df["smoker"].isin(selected_smoker))
            | (df["region"].isin(selected_region))
            | (df["bmi_class"].isin(selected_bmi_class))
        ]
    else:
        # If no options are selected, use the entire dataset
        filtered_data = df.copy()

    # Plot the scatter plot using Matplotlib
    scatter_plot = sns.scatterplot(
        data=filtered_data,
        x=selected_columns[0],
        y=selected_columns[1],
        color=custom_colour,
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
            data=filtered_data,
            x=feature,
            color=custom_colour,
            bins=bin_size,
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

elif plot_type == "scatter" and len(selected_columns) != 2:
    st.warning("Please select exactly two features for the scatter plot.")







from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Sidebar for Model Evaluation options
selected_models = st.sidebar.multiselect(
    "Select Model",
    ["Decision Tree", "Lasso Regression", "Light GBM", "Random Forest", "Ridge Regression", "XGBoost"]
)

# Extract features and target variable
X = df.drop(columns=['charges'])
# Extract target variable
y = df['charges']
# Split the data into training and testing sets with stratification
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df['smoker'])

# On/Off switch for feature importance
show_feature_importance = st.sidebar.checkbox("Show Feature Importance", False)

# Evaluate and plot metrics for the selected models
for selected_model in selected_models:
    st.subheader(f"Model Evaluation: {selected_model}")

    # Get the corresponding model from the trained models
    if selected_model == "Decision Tree":
        model = random_search_decision_tree.best_estimator_
    elif selected_model == "Lasso Regression":
        model = random_search_lasso_regression.best_estimator_
    elif selected_model == "Light GBM":
        model = random_search_lightgbm.best_estimator_
    elif selected_model == "Random Forest":
        model = random_search_random_forest.best_estimator_
    elif selected_model == "Ridge Regression":
        model = random_search_ridge_regression.best_estimator_
    elif selected_model == "XGBoost":
        model = random_search_xgboost.best_estimator_
    else:
        st.warning(f"No model found for {selected_model}")
        continue

    # Fit the model and make predictions
    model.fit(Xtrain, ytrain)
    y_pred = model.predict(Xtest)

    # Calculate metrics
    mae = mean_absolute_error(ytest, y_pred)
    mse = mean_squared_error(ytest, y_pred)
    rmse = np.sqrt(mse)

    # Display metrics
    st.text(f"MAE for {selected_model}: {mae:.2f}")
    st.text(f"MSE for {selected_model}: {mse:.2f}")
    st.text(f"RMSE for {selected_model}: {rmse:.2f}")

    # Plot feature importance if selected
    if show_feature_importance:
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
            features = Xtrain.columns
            plot_data = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
            plot_data = plot_data.sort_values(by='Importance', ascending=False)

            # Plot the feature importance
            st.subheader(f"Feature Importance for {selected_model}")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=plot_data, ax=ax)
            st.pyplot(fig)

    # Plot metrics
    fig_metrics, ax_metrics = plt.subplots(figsize=(10, 6))
    metrics_labels = ['MAE', 'MSE', 'RMSE']
    metrics_values = [mae, mse, rmse]
    ax_metrics.bar(metrics_labels, metrics_values, color=['blue', 'green', 'orange'])
    ax_metrics.set_title(f'Metrics for {selected_model}')
    ax_metrics.set_xlabel('Metrics')
    ax_metrics.set_ylabel('Values')
    st.pyplot(fig_metrics)

    # Display individual values on top of each bar
    for i, v in enumerate(metrics_values):
        ax_metrics.text(i, v + 50, f"{v:.2f}", ha='center', va='bottom')

    st.pyplot(fig_metrics)





