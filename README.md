# PhishGuard AI

A Streamlit demo that classifies URLs as likely legitimate or phishing using URL features and a Random Forest model.

## Features

- URL scanner with risk estimate and URL checks
- Logistic Regression and Random Forest comparison
- Domain-grouped train/test split to reduce domain overlap
- Basic feature-importance display

## Requirements

Python 3.10 or newer. From the project folder, activate your virtual environment and install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

The project uses `ucimlrepo` to download the dataset and `tldextract` for domain-grouped splitting. If they are not listed in `requirements.txt`, install them with:

```powershell
python -m pip install ucimlrepo tldextract
```

## Download data and train

Download the UCI PhiUSIIL Phishing URL Dataset and save the URL and label columns locally:

```powershell
python download_dataset.py
```

Train the models and save their metrics and the Random Forest model:

```powershell
python train.py
```

The dataset is saved as `data/urls.csv` and excluded from Git. The trained model and metrics are saved in `artifacts/`.

## Run the app

```powershell
python -m streamlit run app.py
```

Streamlit will print a local address, usually `http://localhost:8501`.

## Evaluation notes

The project groups URLs by registered domain when splitting training and test data. This helps prevent URLs from the same domain appearing in both sets. The printed precision, recall, F1-score, and accuracy describe performance on this dataset and split only. They do not guarantee detection of new or future phishing URLs.

The app analyzes URL text and does not visit or inspect the website. HTTPS does not prove that a website is legitimate.

## Dataset and labels

Dataset: [UCI PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)

The downloader maps the UCI labels to this project's convention:

- `0` = legitimate
- `1` = phishing

Review the dataset's license and citation requirements before redistributing the data or reporting results.

## Project files

- `app.py` — Streamlit interface
- `features.py` — URL feature extraction
- `train.py` — training and evaluation
- `download_dataset.py` — dataset download and CSV preparation
- `requirements.txt` — Python dependencies
- `artifacts/` — saved model and evaluation metrics
