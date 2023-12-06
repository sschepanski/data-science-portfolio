# Medical Insurance Charges Prediction

## Introduction

This Jupyter Notebook explores the relationship between personal attributes, geographic factors, and their impact on medical insurance charges. The dataset provides valuable insights into how features such as age, gender, BMI, family size, smoking habits, and region influence healthcare expenses. The ultimate goal is to develop predictive models that can estimate medical insurance costs based on these factors.

## Dataset Overview

The dataset comprises the following key features:

- **Age:** The insured person's age.
- **Sex:** Gender (male or female) of the insured.
- **BMI (Body Mass Index):** A measure of body fat based on height and weight.
- **Children:** The number of dependents covered.
- **Smoker:** Whether the insured is a smoker (yes or no).
- **Region:** The geographic area of coverage.
- **Charges:** The medical insurance costs incurred by the insured person.

## Goal

1. **Exploratory Data Analysis (EDA):** Understand the distribution of each feature, identify patterns, and explore potential correlations.

2. **Feature Engineering:** Derive new features or preprocess existing ones to enhance model performance.

3. **Predictive Modeling:** Develop machine learning models to predict medical insurance charges based on the available features.

4. **Model Evaluation:** Assess the performance of the models using relevant metrics and fine-tune as necessary.

## Project Structure

- **`notebooks/`**: Jupyter notebooks for EDA, feature engineering, and model development.
- **`data/`**: Dataset file (e.g., `insurance_data.csv`).
- **`models/`**: Saved models for deployment or further analysis.
- **`reports/`**: Project reports, visualizations, and model evaluation results.

## Getting Started

1. **Clone this repository:**

   ```bash
   git clone https://github.com/your-username/medical-insurance-charges.git

   
2. Use the requirements file in this repository to create a new environment.

```Bash
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

3. Explore the Jupyter notebooks for detailed analysis.

4. Run the provided scripts in the src/ directory for data preprocessing, model training, and evaluation.

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


## **Contributions**

This project was conducted by Dr. Steven Schepanski.

## **License**

This project is licensed under the MIT License.