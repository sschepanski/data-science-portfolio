# Heart Attack Analysis & Prediction

## Introduction

This project focuses on analyzing and predicting the likelihood of a heart attack based on a dataset containing various medical and lifestyle factors. The dataset includes information such as age, sex, exercise-induced angina, chest pain type, resting blood pressure, cholesterol levels, fasting blood sugar, electrocardiographic results, maximum heart rate achieved, and the presence or absence of a heart attack.

## Goal

The primary goal of this project is to utilize machine learning techniques to analyze the dataset and build a predictive model for heart attack classification. By exploring the relationships between various features and the target variable, we aim to develop a model that can accurately predict the likelihood of a heart attack based on the provided data.

## Project Structure

- **`data/`**: Dataset files (e.g., `heart.csv`).
- **`models/`**: Saved models for deployment or further analysis.
- **`scr/`**: Saved Jupyter notebooks for data exploration, preprocessing, model training, and evaluation.
- **Dashboard.py**: Streamlit dashboard for interactive visualisation and exploration.

## Analysis

1. **Data Exploration:** Understanding the dataset's structure, exploring its intricacies, and examining the distribution of variables.
2. **Data Preprocessing:** Rigorous cleaning procedures, addressing missing values, and preparing the data for subsequent modeling stages.
3. **Exploratory Data Analysis (EDA):** Visualizing relationships between variables and extracting meaningful patterns to gain insights.
4. **Feature Selection:** Identifying and selecting the most relevant features crucial for our predictive task.
5. **Model Selection and Training:** Choosing suitable machine learning algorithms tailored to our classification objective, and training and refining the selected models to optimize performance.
6. **Model Evaluation:** Thoroughly assessing the efficacy of the models using diverse evaluation metrics.
7. **Fine-Tuning:** Interatively refining the models to optimise performance.
8. **Feature Importance:** Analyzing the importance of features in the predictive model to understand which variables contribute significantly to the heart attack prediction.
9. **Conclusion:** A concise summary encapsulating our findings and an evaluation of the predictive model's effectiveness.

## Data Dictionary

| Feature   | Description |
|-----------|-------------|
| `Age`     | Age of the patient |
| `Sex`     | Sex of the patient |
| `exang`   | Exercise-induced angina (1 = yes; 0 = no) |
| `ca`      | Number of major vessels (0-3) |
| `cp`      | Chest pain type |
| `trtbps`  | Resting blood pressure (in mm Hg) |
| `chol`    | Cholesterol in mg/dl fetched via BMI sensor |
| `fbs`     | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) |
| `rest_ecg`| Resting electrocardiographic results |
| `thalach` | Maximum heart rate achieved |
| `target`  | 0= less chance of heart attack, 1= more chance of heart attack |

## Acknowledgments

Dataset source: [Heart Attack Analysis & Prediction Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset/data)

## Getting Started

1. **Clone this repository:**

   ```bash
   git clone https://github.com/sschepanski/data-science-portfolio.git
   ```

2. **Set up your enviornment using the provided requirement file:**
   ```bash
   pyenv local 3.11.3
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Run the provided notebooks in the `scr/` directory for data preprocessing, model training, and evaluation.**

## Contributions

This project was conducted by Dr. Steven Schepanski.

## License

This project is licensed under the MIT License.