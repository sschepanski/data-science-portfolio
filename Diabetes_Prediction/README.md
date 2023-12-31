# **Predicting Diabetes Onset**

## **Introduction**

This project aims to develop a machine learning model to predict the likelihood of diabetes onset in patients using a Diabetes Database. The dataset comprises various medical predictor variables.

## **Goal**

The primary objective is to utilise these features to achieve an accurate prediction of diabetes onset.

## **Project Structure**

- **`data/`**: Dataset files (e.g., `diabetes_data.csv`).
- **`models/`**: Saved models for deployment or further analysis.
- **`scr/`**: Saved Jupyter notebooks for data exploration, preprocessing, model training, and evaluation.

## **Analysis**

1. **Data Exploration:** Delve into the dataset's intricacies, understand its structure, and scrutinise the distribution of variables.
2. **Data Preprocessing:** Implement rigorous cleaning procedures, address missing values, and prepare the data meticulously for subsequent modelling stages.
3. **Exploratory Data Analysis (EDA):** Unveil insights by visually depicting relationships between variables and extracting meaningful patterns.
4. **Feature Selection:** Discern the most pertinent features crucial for our predictive task.
5. **Model Selection and Training:** Identify and choose the most suitable machine learning algorithms tailored to our classification objective. Train and refine the chosen models to optimise performance.
6. **Model Evaluation:** Rigorously assess model efficacy using diverse evaluation metrics.
7. **Fine-Tuning:** Interatively refining the models to optimise performance.
8. **Conclusion:** Provide a succinct summation encapsulating our discoveries and an evaluation of the predictive model's effectiveness.

## **Data Dictionary**

| Feature                     | Description                                                |
|-----------------------------|------------------------------------------------------------|
| `Pregnancies`               | Number of pregnancies.                                     |
| `Glucose`                   | Plasma glucose concentration after a 2-hour oral glucose tolerance test. |
| `BloodPressure`             | Diastolic blood pressure (mm Hg).                          |
| `SkinThickness`             | Triceps skinfold thickness (mm).                            |
| `Insulin`                   | 2-hour serum insulin (mu U/ml).                            |
| `BMI`                       | Body mass index (weight in kg/(height in m)^2).            |
| `DiabetesPedigreeFunction`  | A function that quantifies diabetes history in relatives.  |
| `Age`                       | Age in years.                                              |
| `Outcome`                   | Target variable; 1 if the patient has diabetes, 0 otherwise.|

## Acknowledgements

Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. In Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261--265). IEEE Computer Society Press.

## **Getting Started**

1. **Clone this repository:**

   ```bash
   git clone https://github.com/sschepanski/data-science-portfolio.git
   
2. Use the requirements file in this repository to create a new environment.

   ```Bash
   pyenv local 3.11.3
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Run the provided notebooks in the `scr/` directory for data preprocessing, model training, and evaluation.**

## **Contributions**

This project is a collaborative effort involving contributions from different individuals, including Dr. Steven Schepanski and other attendees of the SPICED Academy Data Science Bootcamp.

## **License**

This project is licensed under the MIT License.