# Stroke Prediction

## Introduction

This project focuses on developing a machine learning model for predicting the likelihood of stroke events based on 11 clinical features. The dataset used for this analysis is the **Stroke Prediction Dataset**, which includes essential information to predict whether a patient is likely to experience a stroke. Stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths, according to the World Health Organization (WHO).

## Goal

Our primary goal is to leverage the provided clinical features, such as gender, age, various diseases, and smoking status, to create an accurate predictive model for identifying patients at risk of stroke.

## Project Structure

- **`notebooks/`**: Jupyter notebooks for data exploration, preprocessing, model training, and evaluation.
- **`data/`**: Dataset files (e.g., `healthcare-dataset-stroke-data.csv`).
- **`reports/`**: Project reports, visualizations, and results.

## Analysis

1. **Data Exploration:** Understanding the dataset's structure, exploring its intricacies, and examining the distribution of clinical features.
2. **Data Preprocessing:** Rigorous cleaning procedures, addressing missing values, and preparing the data for subsequent modeling stages.
3. **Exploratory Data Analysis (EDA):** Visualizing relationships between clinical features and extracting meaningful patterns to gain insights.
4. **Feature Selection:** Identifying and selecting the most relevant clinical features crucial for our predictive task.
5. **Model Selection and Training:** Choosing suitable machine learning algorithms tailored to our classification objective, and training and refining the selected models to optimize performance.
6. **Model Evaluation:** Thoroughly assessing the efficacy of the models using diverse evaluation metrics.
7. **Feature Importance:** Analyzing the importance of clinical features in the predictive model to understand which variables contribute significantly to stroke prediction.
8. **Conclusion:** A concise summary encapsulating our findings and an evaluation of the predictive model's effectiveness.

## Data Dictionary

| Feature              | Description                                               |
|----------------------|-----------------------------------------------------------|
| `id`                 | Unique identifier.                                        |
| `gender`             | "Male", "Female" or "Other".                              |
| `age`                | Age of the patient.                                       |
| `hypertension`       | 0 if the patient doesn't have hypertension, 1 if the patient has hypertension. |
| `heart_disease`      | 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease. |
| `ever_married`       | "No" or "Yes".                                           |
| `work_type`          | "children", "Govt_jov", "Never_worked", "Private" or "Self-employed". |
| `Residence_type`     | "Rural" or "Urban".                                      |
| `avg_glucose_level`  | Average glucose level in blood.                           |
| `bmi`                | Body mass index.                                          |
| `smoking_status`     | "formerly smoked", "never smoked", "smokes" or "Unknown" (unavailable information). |
| `stroke`             | 1 if the patient had a stroke, 0 if not.                 |

## Acknowledgements

(Confidential Source) - Use only for educational purposes. If you use this dataset in your research, please credit the author.

## Getting Started

1. **Clone this repository:**

   ```bash
   git clone https://github.com/your-username/predicting-kickstarter.git

2. **Set up your environment using the provided requirements file:**
   ```bash
   pyenv local 3.11.3
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Explore the Jupyter notebooks for detailed analysis.**
   
4. **Run the provided scripts in the src/ directory for data preprocessing, model training, and evaluation.**

## **Contributions**

This project is a collaborative effort involving contributions from different individuals, including Dr. Steven Schepanski and other attendees of the SPICED Academy Data Science Bootcamp.

**## License**

This project is licensed under the MIT License.