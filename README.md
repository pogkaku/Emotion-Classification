# Emotion Classification Using Machine Learning

An NLP-based machine learning project that classifies text into **seven emotion categories** using text preprocessing, TF-IDF feature extraction, and supervised machine learning.

## Project Overview

The goal of this project is to build and evaluate machine learning models capable of identifying the emotion expressed in a piece of text.

The project compares three classification algorithms:

- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (Linear SVM)

The best-performing model was **Linear SVM**, achieving approximately **91.19% accuracy** on the held-out test set.

## Dataset

The dataset contains **8,000 text samples** with two main columns:

- `text` — the input text
- `emotion` — the target emotion label

The dataset contains seven emotion classes:

- `fun`
- `happiness`
- `hate`
- `neutral`
- `sadness`
- `surprise`
- `worry`

The data was split using an **80/20 stratified train-test split**:

- Training set: 6,400 samples
- Test set: 1,600 samples

Stratification was used to preserve the distribution of the emotion classes across the training and test sets.

## Data Preprocessing

The text preprocessing pipeline includes:

1. Converting text to lowercase
2. Removing URLs
3. Removing numbers
4. Removing punctuation and special characters
5. Removing extra whitespace
6. Removing English stopwords
7. Lemmatizing words using NLTK WordNet

These steps reduce noise and create a more consistent representation of the text for machine learning.

> **Note:** The current preprocessing function keeps Latin/English characters (`a-zA-Z`). If the dataset is intended to support multilingual text, this step should be revised rather than removing non-Latin characters.

## Feature Engineering

Text was converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

Configuration:

- Maximum features: `10,000`
- N-gram range: `(1, 2)`
- Unigrams: individual words
- Bigrams: two-word combinations

Bigrams are useful because short phrases can carry emotional meaning that may not be captured by individual words alone.

## Models Evaluated

### 1. Multinomial Naive Bayes

Used as the baseline classifier.

**Test accuracy: approximately 79.50%**

### 2. Logistic Regression

Provided a substantial improvement over the Naive Bayes baseline.

**Test accuracy: approximately 90.38%**

### 3. Linear SVM

The Linear SVM produced the strongest overall performance.

**Test accuracy: approximately 91.19%**

| Model | Accuracy |
|---|---:|
| Multinomial Naive Bayes | 79.50% |
| Logistic Regression | 90.38% |
| Linear SVM | **91.19%** |

## Model Validation and Tuning

Cross-validation and GridSearchCV were used to evaluate and tune the Linear SVM.

The hyperparameter optimized was:

```text
C
```

The final model uses:

```python
LinearSVC(C=1)
```

The notebook uses **9-fold cross-validation** for model selection and evaluation.

## Evaluation

The project evaluates the models using:

- Accuracy
- Precision
- Recall
- F1-score
- Classification reports
- Confusion matrix
- Cross-validation
- Hyperparameter tuning

The confusion matrix shows that most predictions fall on the diagonal, indicating strong classification performance. Some overlap remains between emotionally similar categories such as `hate` and `worry`, and `sadness` and `hate`.

## Project Structure

```text
Emotion-Classification/
│
├── emotion-classification-using-machine-learning.ipynb
├── main.py
├── requirements.txt
└── README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/pogkaku/Emotion-Classification.git
cd Emotion-Classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

Open:

```text
emotion-classification-using-machine-learning.ipynb
```

The notebook contains the complete exploratory analysis, preprocessing, model training, evaluation, and tuning workflow.

## Using `main.py`

`main.py` provides a reusable prediction pipeline.

Your training CSV should contain:

```text
text,emotion
```

Your prediction/test CSV should contain:

```text
text
```

Run:

```bash
python main.py --train train.csv --test test.csv --output predictions.csv
```

The script will:

1. Load the datasets
2. Clean the text
3. Fit a TF-IDF vectorizer
4. Train the Linear SVM
5. Generate predictions
6. Save the predictions to a CSV file

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn
- Jupyter Notebook
- Kaggle

## Key Learning Outcomes

This project demonstrates practical experience with:

- Natural Language Processing (NLP)
- Text preprocessing
- Exploratory data analysis
- TF-IDF feature engineering
- Multi-class classification
- Model comparison
- Cross-validation
- Hyperparameter tuning
- Confusion-matrix analysis
- Building a reusable inference pipeline

## Future Improvements

Possible improvements include:

- Testing transformer-based models such as BERT
- Improving multilingual preprocessing
- Handling class imbalance with additional techniques
- Performing more extensive hyperparameter optimization
- Saving the trained vectorizer and model for deployment
- Creating a web API or interactive application for real-time emotion prediction

## Author

**Ghislain Akama Nso**

Machine Learning / Data Science Portfolio Project
