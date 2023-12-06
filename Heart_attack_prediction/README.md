# **Heart Attack Analysis & Prediction**

## **Introduction**

This project focuses on analyzing and predicting the likelihood of a heart attack based on a dataset containing various medical and lifestyle factors. The dataset includes information such as age, sex, exercise-induced angina, chest pain type, resting blood pressure, cholesterol levels, fasting blood sugar, electrocardiographic results, maximum heart rate achieved, and the presence or absence of a heart attack.

## **Goal**

The primary goal of this project is to utilize machine learning techniques to analyze the dataset and build a predictive model for heart attack classification. By exploring the relationships between various features and the target variable, we aim to develop a model that can accurately predict the likelihood of a heart attack based on the provided data.

## **Project Structure**

- **`notebooks/`**: Jupyter notebooks for data exploration, preprocessing, model training, and evaluation.
- **`data/`**: Dataset files (e.g., `heart_attack_data.csv`).
- **`reports/`**: Project reports and visualizations.

## **Data Dictionary**

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

## **Analysis Steps**

1. **Data Exploration:** Understand the dataset's structure, examine feature distributions, and identify patterns.

2. **Data Preprocessing:** Clean the data, handle missing values, and prepare it for modeling.

3. **Exploratory Data Analysis (EDA):** Visualize relationships between variables and extract meaningful insights.

4. **Feature Selection:** Identify key features for predictive modeling.

5. **Model Selection and Training:** Choose suitable machine learning algorithms and train models for heart attack prediction.

6. **Model Evaluation:** Evaluate model performance using various metrics.

7. **Conclusion:** Summarize findings and assess the effectiveness of the predictive model.

## **Getting Started**

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/heart-attack-analysis.git
   ```

2. Set up a virtual environment and install dependencies:
   ```bash
   pyenv local 3.11.3
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Explore Jupyter notebooks for detailed analysis.
4. Run scripts in the src/ directory for data preprocessing, model training, and evaluation.

## **Contributions**

This project welcomes contributions from the community to enhance its analysis and predictive capabilities.

## **License**

This project is licensed under the MIT License.