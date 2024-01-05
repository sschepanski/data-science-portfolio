# Natural Language Processing with Disaster Tweets

## Introduction

This project focuses on Natural Language Processing (NLP) to predict whether tweets are about real disasters or not. The dataset, provided by Figure-Eight and shared on Kaggle, consists of 10,000 tweets that have been hand-classified. The goal is to build a machine learning model capable of distinguishing between tweets describing real disasters and those that do not.

## Goal

The primary objective is to leverage NLP techniques to develop an accurate predictive model for classifying tweets. This has practical applications for organizations involved in disaster relief and news agencies seeking to programmatically monitor Twitter for real-time emergency announcements.

## Project Structure

- **`data/`**: Dataset files (e.g., `tweets_dataset.csv`).
- **`models/`**: Saved models for deployment or further analysis.
- **`scr/`**: Saved Jupyter notebooks for data exploration, preprocessing, model training, and evaluation.
- **Dashboard.py**: Streamlit dashboard for interactive visualisation and exploration.

## Analysis

1. **Data Exploration:** Understanding the structure of the dataset and exploring tweet content and associated labels.
2. **Data Preprocessing:** Cleaning and preparing the text data for NLP tasks, handling profanity, and addressing missing values.
3. **Text Tokenization:** Breaking down the tweets into tokens to enable machine understanding of the textual information.
4. **Exploratory Data Analysis (EDA):** Visualising tweet patterns, word frequencies, and exploring relationships between language and disaster classification.
5. **Model Selection and Training:** Choosing suitable NLP models, such as recurrent neural networks (RNNs) or transformer-based models, and training them to classify tweets.
6. **Model Evaluation:** Assessing model performance using metrics like F1 score, precision, and recall.
7. **Fine-Tuning:** Iteratively refining the models to optimise performance.
8. **Conclusion:** Summarising findings, discussing challenges, and evaluating the model's effectiveness in predicting disaster tweets.

## Data Dictionary

| Feature                     | Description                                                |
|-----------------------------|------------------------------------------------------------|
| `id`               | Unique identifier for each tweet. |
| `target`           | Binary label (0 or 1) indicating whether the tweet is describing a real disaster. |

## Acknowledgements

This dataset was originally shared on Figure-Eight's 'Data For Everyone' website. The competition is hosted on Kaggle to encourage data scientists to explore NLP and engage in a beginner-friendly competition.

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