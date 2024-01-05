# Medical Insurance Charges Prediction

## Introduction

This Jupyter Notebook explores the relationship between personal attributes, geographic factors, and their impact on medical insurance charges. The dataset provides valuable insights into how features such as age, gender, BMI, family size, smoking habits, and region influence healthcare expenses. The ultimate goal is to develop predictive models that can estimate medical insurance costs based on these factors.

## Goal

The goal of this project is to develop predictive models that can estimate health insurance costs based on lifestyle factors. Using these models, a potential insurance company can estimate whether it is financially viable to insure/sign new customers.

## Project Structure

- **`data/`**: Dataset files (e.g., `insurance.csv`).
- **`models/`**: Saved models for deployment or further analysis.
- **`scr/`**: Saved Jupyter notebooks for data exploration, preprocessing, model training, and evaluation.
- **Dashboard.py**: Streamlit dashboard for interactive visualisation and exploration.

## Analysis

1. **Data Exploration:** Delve into the dataset's intricacies, understand its structure, and scrutinise the distribution of variables.
2. **Data Preprocessing:** Implement rigorous cleaning procedures, address missing values, and prepare the data meticulously for subsequent modelling stages.
3. **Exploratory Data Analysis (EDA):** Unveil insights by visually depicting relationships between variables and extracting meaningful patterns.
4. **Feature Engineering:** Derive new features or preprocess existing ones to enhance model performance.
5. **Model Selection and Training:** Develop machine learning models to predict medical insurance charges based on the available features.
6. **Model Evaluation:** Assess the performance of the models using relevant metrics and fine-tune as necessary.
7. **Fine-Tuning:** Interatively refining the models to optimise performance.
8. **Feature Importance:** Analyse the importance of features in the predictive model to understand which factors contribute significantly to medical insurance charges.
9. **Conclusion:** Provide a succinct summation encapsulating our discoveries and an evaluation of the predictive model's effectiveness.

## Data Dictionary

| Feature   | Description                                             |
|-----------|---------------------------------------------------------|
| `Age`     | The age of the insured person.                           |
| `Sex`     | Gender of the insured (male or female).                  |
| `BMI`     | Body Mass Index: a measure of body fat based on height and weight.|
| `Children`| The number of dependents covered by the insurance.       |
| `Smoker`  | Whether the insured is a smoker (yes or no).            |
| `Region`  | The geographic area of coverage.                         |
| `Charges` | Medical insurance costs incurred by the insured person.  |

## Acknowledgements

Dataset source: [Healthcare Insurance Dataset](https://www.kaggle.com/datasets/willianoliveiragibin/healthcare-insurance/data)

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

## **Contributions**

This project was conducted by Dr. Steven Schepanski.

## **License**

This project is licensed under the MIT License.