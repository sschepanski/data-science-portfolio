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

# Average Charges card
average_charges_card = (
    f"<div style='border: 1px solid #ccc; border-radius: 10px; padding: 10px; background-color: #cacccd;'>"
    f"<div style='text-align: center; color: #f5f5f5;'><b>Average Charges</b></div>"
    f"<div style='text-align: center; font-size: 36px;'>{average_charges_all:.2f}</div>"
    f"<div style='display: flex; justify-content: space-around; margin-top: 10px;'>"
    f"   <div style='font-size: 18px; color: #15556f;'>♂{average_charges_male:.2f}</div>"
    f"   <div style='font-size: 18px; color: #FF7E00;'>♀{average_charges_female:.2f}</div>"
    f"</div>"
    f"</div>"
)

# Average Age card
average_age_card = (
    f"<div style='border: 1px solid #ccc; border-radius: 10px; padding: 10px; background-color: #cacccd;'>"
    f"<div style='text-align: center; color: #f5f5f5;'><b>Average Age</b></div>"
    f"<div style='text-align: center; font-size: 36px;'>{average_age_all:.2f}</div>"
    f"<div style='display: flex; justify-content: space-around; margin-top: 10px;'>"
    f"   <div style='font-size: 18px; color: #15556f;'>♂{average_age_male:.2f}</div>"
    f"   <div style='font-size: 18px; color: #FF7E00;'>♀{average_age_female:.2f}</div>"
    f"</div>"
    f"</div>"
)

# Average BMI card
average_bmi_card = (
    f"<div style='border: 1px solid #ccc; border-radius: 10px; padding: 10px; background-color: #cacccd;'>"
    f"<div style='text-align: center; color: #f5f5f5;'><b>Average BMI</b></div>"
    f"<div style='text-align: center; font-size: 36px;'>{average_bmi_all:.2f}</div>"
    f"<div style='display: flex; justify-content: space-around; margin-top: 10px;'>"
    f"   <div style='font-size: 18px; color: #15556f;'>♂{average_bmi_male:.2f}</div>"
    f"   <div style='font-size: 18px; color: #FF7E00;'>♀{average_bmi_female:.2f}</div>"
    f"</div>"
    f"</div>"
)

# Define the three columns
columns = st.columns(3)

# Average Charges card
columns[0].markdown(average_charges_card, unsafe_allow_html=True)

# Average Age card
columns[1].markdown(average_age_card, unsafe_allow_html=True)

# Average BMI card
columns[2].markdown(average_bmi_card, unsafe_allow_html=True)

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

# Mapping for sex
sex_mapping = {0: "male", 1: "female"}
sex_options = list(sex_mapping.values())
sex_options.append("All")
selected_sex = st.sidebar.radio("Select Sex", sex_options, index=len(sex_options) - 1)
selected_sex_code = [key for key, value in sex_mapping.items() if value == selected_sex][0] if selected_sex != "All" else None

# Mapping for smoker
smoker_mapping = {0: "non-smoker", 1: "smoker"}
smoker_options = list(smoker_mapping.values())
smoker_options.append("All")
selected_smoker = st.sidebar.radio("Select Smoker", smoker_options, index=len(smoker_options) - 1)
selected_smoker_code = [key for key, value in smoker_mapping.items() if value == selected_smoker][0] if selected_smoker != "All" else None

# Mapping for region
region_mapping = {"southwest": "Southwest", "southeast": "Southeast", "northwest": "Northwest", "northeast": "Northeast"}
region_options = list(region_mapping.values())
region_options.append("All")
selected_region = st.sidebar.radio("Select Region", region_options, index=len(region_options) - 1)
selected_region_code = [key for key, value in region_mapping.items() if value == selected_region][0] if selected_region != "All" else None

# Mapping for bmi_class
bmi_class_mapping = {1: "underweight", 2: "healthy", 3: "overweight", 4: "obese"}
bmi_class_options = list(bmi_class_mapping.values())
bmi_class_options.append("All")
selected_bmi_class = st.sidebar.radio("Select BMI Class", bmi_class_options, index=len(bmi_class_options) - 1)
selected_bmi_class_code = [key for key, value in bmi_class_mapping.items() if value == selected_bmi_class][0] if selected_bmi_class != "All" else None

# Choose the type of plot
plot_type = st.sidebar.selectbox("Select Plot Type", ["scatter"])

# Choose color
custom_colour = (108 / 255, 84 / 255, 158 / 255)  # RGB(108, 84, 158)

# Scatter plot based on user selection
if plot_type == "scatter" and len(selected_columns) == 2:
    # Show selected plot_type in the second line as the first column
    st.subheader(f"Selected Plot Type: {plot_type}")

    # Create a Matplotlib Figure for the scatter plot with a larger size
    fig, (ax_scatter, ax_hist1, ax_hist2) = plt.subplots(1, 3, figsize=(25, 5))

    # Filter data based on selected options
    if selected_sex_code is not None:
        df = df[df["sex"] == selected_sex_code]
    if selected_smoker_code is not None:
        df = df[df["smoker"] == selected_smoker_code]
    if selected_region_code is not None:
        df = df[df["region"] == selected_region_code]
    if selected_bmi_class_code is not None:
        df = df[df["bmi_class"] == selected_bmi_class_code]

    # Plot the scatter plot using Matplotlib
    scatter_plot = sns.scatterplot(
        data=df,
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
            data=df,
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

# Create three separate columns for MAE, MSE, and RMSE cards
columns_metrics = st.columns(3)

# Initialize metrics outside the loop
metrics_data = {}

# Evaluate and plot metrics for the selected models
show_subheader = True  # Flag to control the display of subheaders

for selected_model in selected_models:
    if show_subheader:
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

    # Store metrics in the dictionary
    metrics_data[selected_model] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse}

    # If more than one model is selected, set the flag to False
    if len(selected_models) > 1:
        show_subheader = False

# Display cards for MAE, MSE, and RMSE only if one model is selected
if len(selected_models) == 1:
    selected_model = selected_models[0]

    # MAE card
    mae_card = (
        f"<div style='border: 1px solid #ccc; border-radius: 10px; padding: 10px; background-color: #cacccd;'>"
        f"<div style='text-align: center; color: #f5f5f5;'><b>MAE</b></div>"
        f"<div style='text-align: center; font-size: 36px;'>{metrics_data[selected_model]['MAE']:.2f}</div>"
        "</div>"
    )

    # MSE card
    mse_card = (
        f"<div style='border: 1px solid #ccc; border-radius: 10px; padding: 10px; background-color: #cacccd;'>"
        f"<div style='text-align: center; color: #f5f5f5;'><b>MSE</b></div>"
        f"<div style='text-align: center; font-size: 36px;'>{metrics_data[selected_model]['MSE']:.2f}</div>"
        "</div>"
    )

    # RMSE card
    rmse_card = (
        f"<div style='border: 1px solid #ccc; border-radius: 10px; padding: 10px; background-color: #cacccd;'>"
        f"<div style='text-align: center; color: #f5f5f5;'><b>RMSE</b></div>"
        f"<div style='text-align: center; font-size: 36px;'>{metrics_data[selected_model]['RMSE']:.2f}</div>"
        "</div>"
    )

    # Define the three columns
    columns = st.columns(3)

    # Display MAE card
    columns[0].markdown(mae_card, unsafe_allow_html=True)

    # Display MSE card
    columns[1].markdown(mse_card, unsafe_allow_html=True)

    # Display RMSE card
    columns[2].markdown(rmse_card, unsafe_allow_html=True)

    # Add a separator
    st.markdown("<hr style='margin: 20px;'>", unsafe_allow_html=True)

# Display horizontal bar charts for MAE, MSE, and RMSE
if len(selected_models) >= 1:  # Adjust the condition to show charts even when multiple models are selected
    st.subheader("Model Evaluation")

    # Display horizontal bar charts for MAE, MSE, and RMSE
    fig_metrics, ax_metrics = plt.subplots(1, 3, figsize=(18, 6))  # Change subplot dimensions for a single row

    # Create horizontal bar charts for MAE, MSE, and RMSE
    metrics_labels = ['MAE', 'MSE', 'RMSE']

    # Create a dictionary to store the color for each model
    model_colors = dict(zip(selected_models, plt.cm.viridis(np.linspace(0, 1, len(selected_models)))))

    for i, metric in enumerate(metrics_labels):
        # Add bars for each selected model, check if model exists in metrics_data
        bars = [metrics_data[model][metric] if model in metrics_data else 0 for model in selected_models]

        # Assign different colors to each bar
        colors = [model_colors[model] for model in selected_models]

        ax_metrics[i].bar(selected_models, bars, color=colors, label=metric)
        ax_metrics[i].set_title(f'{metric}')
        ax_metrics[i].set_ylabel(metric)
        
        # Hide x-axis labels
        ax_metrics[i].set_xticks([])

    # Create a single legend for the entire figure
    legend_colors = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=model_colors[model], markersize=10, label=model) for model in selected_models]
    fig_metrics.legend(handles=legend_colors, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=len(selected_models))

    # Display the entire subplot in a single row
    st.pyplot(fig_metrics)



# Enable the "Show Feature Importance" checkbox only for tree-based models
tree_models = ["Decision Tree", "Random Forest", "Light GBM", "XGBoost"]

# Check if at least one tree-based model is selected
show_feature_importance_checkbox = any(model in tree_models for model in selected_models)

# On/Off switch for feature importance if at least one tree-based model is selected
show_feature_importance = st.sidebar.checkbox("Show Feature Importance", show_feature_importance_checkbox)

# Load the best models and configurations
best_decision_tree_model = random_search_decision_tree.best_estimator_
best_lightgbm_model = random_search_lightgbm.best_estimator_
best_random_forest_model = random_search_random_forest.best_estimator_
best_xgboost_model = random_search_xgboost.best_estimator_

# If feature importance is enabled and at least one model is selected
if show_feature_importance and selected_models:
    st.subheader("Feature Importance")

    for selected_model in selected_models:
        # Check if the selected model is a tree-based model
        if selected_model in tree_models:
            st.subheader(f"Feature Importance for {selected_model}")

            # Get the corresponding model from the loaded best models
            if selected_model == "Decision Tree":
                model = best_decision_tree_model.named_steps['model']
            elif selected_model == "Random Forest":
                model = best_random_forest_model.named_steps['model']
            elif selected_model == "Light GBM":
                model = best_lightgbm_model.named_steps['model']
            elif selected_model == "XGBoost":
                model = best_xgboost_model.named_steps['model']
            else:
                st.warning(f"No feature importance available for {selected_model}")
                continue

            feature_importances = model.feature_importances_

            # If the model has a ColumnTransformer (ct), you can attempt to get feature names
            if hasattr(model, 'named_steps') and 'ct' in model.named_steps:
                transformed_feature_names = model.named_steps['ct'].get_feature_names_out()
            else:
                transformed_feature_names = None

            if transformed_feature_names is not None:
                # Combine one-hot-encoded features into a single feature
                combined_importances = []
                for feature in transformed_feature_names:
                    if 'onehot_encode' in feature:
                        base_feature = feature.split('__')[1]  # Extract the original feature name
                        combined_importances.append((base_feature, feature_importances[transformed_feature_names == feature].sum()))
                    else:
                        combined_importances.append((feature, feature_importances[transformed_feature_names == feature][0]))

                # Create a DataFrame for combined feature importances
                combined_importances_df = pd.DataFrame(combined_importances, columns=['Feature', 'Importance'])

                # Sort the DataFrame by importance in descending order
                combined_importances_df = combined_importances_df.sort_values(by='Importance', ascending=False)

                # Plot combined feature importance for all features in reversed order
                plt.figure(figsize=(12, 8))
                plt.barh(combined_importances_df['Feature'][::-1], combined_importances_df['Importance'][::-1], color='skyblue')
                plt.xlabel('Feature Importance')
                plt.title(f'{selected_model} Feature Importance')
                plt.show()
            else:
                st.warning(f"No feature importance available for {selected_model}")

        else:
            st.warning(f"Feature importance is not available for {selected_model}. It is only supported for tree-based models.")
