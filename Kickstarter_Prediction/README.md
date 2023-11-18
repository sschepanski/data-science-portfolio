# Kickstarter Predictions

## Introduction

This project focuses on developing predictive models for Kickstarter campaigns, aiming to forecast project success or failure based on various factors. The dataset spans a diverse range of Kickstarter projects, providing valuable insights for creators and backers.

## Goal

The primary objective is to build models that can predict the success of a Kickstarter campaign and estimate the number of backers required for a successful campaign.

## Project Structure

- **`notebooks/`**: Jupyter notebooks for data exploration, preprocessing, model training, and evaluation.
- **`data/`**: Dataset files (e.g., `kickstarter_data.csv`).
- **`reports/`**: Project reports, visualizations, and results.

## Analysis Steps

1. **Data Exploration:** Delve into the intricacies of the Kickstarter dataset, understanding its structure and scrutinizing variable distributions.

2. **Data Preprocessing:** Implement thorough cleaning procedures, address missing values, and prepare the data for subsequent modeling stages.

3. **Exploratory Data Analysis (EDA):** Uncover insights by visually depicting relationships between variables and extracting meaningful patterns.

4. **Feature Selection:** Identify the most relevant features crucial for predicting Kickstarter campaign outcomes.

5. **Model Selection and Training:** Choose suitable machine learning algorithms tailored to the classification objective. Train and refine models to optimize performance.

6. **Model Evaluation:** Rigorously assess model efficacy using diverse evaluation metrics.

7. **Conclusion:** Provide a concise summary encapsulating discoveries and an evaluation of the predictive models' effectiveness.

## Data Dictionary

| Column        | Description                                                   |
|---------------|---------------------------------------------------------------|
| `ID`          | Project ID on Kickstarter                                     |
| `Name`        | Project Name on Kickstarter                                   |
| `Category`    | Project Category (Music, Film & Video, ...)                   |
| `Subcategory` | Project Subcategory (Rock, Pop, ...)                          |
| `Country`     | Country of Project/Product Origin                             |
| `Launched`    | Date of Project Launch                                        |
| `Deadline`    | Deadline for Crowd Funding                                    |
| `Goal`        | Amount of Money Needed for Creator to Complete Project in USD |
| `Pledged`     | Amount of Money That Was Pledged/Collected in USD             |
| `Backers`     | Amount of People That Pledged                                 |
| `State`       | State of Completion/Success                                   |

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