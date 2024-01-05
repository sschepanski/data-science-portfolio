# Sales Forecasting Project

## Introduction

This project focuses on sales forecasting for the fictional 'BFCC AG' corporation. The dataset includes sales figures for several customers, and the goal is to prepare the data, predict future sales, and analyze the impact of a fictional 'afo Business Climate Index' on the sales forecast.

## Goal

The primary goal of this project is to orchestrate a seamless flow of tasks to enhance sales forecasting accuracy and insight. This involves leveraging the monthly sales figures to predict the next 6 months accurately. This uncovers the potential influence of external factors on sales predictions.

## Project Structure

- **`data/`**: Dataset files (e.g.,`daten_kunden.csv`).
- **`models/`**: Saved models for deployment or further analysis.
- **`scr/`**: Saved Jupyter notebooks for data exploration, preprocessing, model training, and evaluation.
- **Dashboard.py**: Streamlit dashboard for interactive visualisation and exploration.

## Analysis

1. **Data Exploration:** Understanding the dataset's structure, exploring its intricacies, and examining the distribution of variables.
2. **Data Preprocessing:** Rigorous cleaning procedures, addressing missing values, and preparing the data for subsequent modeling stages.
3. **Exploratory Data Analysis (EDA):** Visualizing relationships between variables and extracting meaningful patterns to gain insights.
4. **Feature Selection:** Identifying and selecting the most relevant features crucial for our predictive task.
5. **Sales Forecasting:** Using historical sales data to predict future values, illustrating the results graphically.
6. **Impact Analysis:** Assessing the impact of changes in the 'afo Business Climate Index' on the sales forecast.
7. **Conclusion:** A concise summary encapsulating our findings and an evaluation of the predictive model's effectiveness.

## Data Dictionary

| Feature   | Description |
|-----------|-------------|
| `daten_kunden.csv`     | Sales data for 'BFCC AG' customers |
| `afo.csv`     | Monthly values of the 'afo Business Climate Index |

## Acknowledgments

Data source: Provided by 'BFCC AG.'

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

This project is a collaborative effort involving contributions from different individuals, including Dr. Steven Schepanski and other attendees of the SPICED Academy Data Science Bootcamp.

## License

This project is licensed under the MIT License.