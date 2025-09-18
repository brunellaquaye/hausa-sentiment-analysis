# Hausa Sentiment Analysis

This repository provides a reproducible pipeline for Hausa sentiment analysis using classical machine learning models and robust text preprocessing.

## Folder Structure

- `hausa_preprocess.py` — Hausa text preprocessing class
- `hausa_preprocess.ipynb` — Data cleaning and preprocessing notebook
- `hausa_train.ipynb` — Main training notebook (run this to reproduce results)
- `hausa_eval.ipynb` — Model evaluation notebook
- `data/` — Contains all required CSV data splits
- `models/hausa_sentiment/` — Saved models and training metadata
- `reports/hausa_sentiment/` — Evaluation results

## Setup Instructions

1. **Clone the repository:**
   ```sh
   git clone https://github.com/YOUR_USERNAME/hausa-sentiment-analysis.git
   cd hausa-sentiment-analysis
   ```

2. **Install dependencies:**
   - Recommended: Use a virtual environment.
   - Install all required packages:
     ```sh
     pip install -r requirements.txt
     ```
   - Or, for conda users:
     ```sh
     conda env create -f environment.yml
     conda activate hausa-sentiment
     ```

3. **Data:**
   - Cleaned CSVs are already included in `data/`.
   - If you need to re-download raw data, run:
     ```sh
     python data/afrisenti_twitter_hausa_loader.py
     ```
   - Dataset source: [HausaNLP/AfriSenti-Twitter on Hugging Face](https://huggingface.co/datasets/HausaNLP/AfriSenti-Twitter)

## How to Reproduce Results

1. **Preprocessing (optional):**
   - Open and run `hausa_preprocess.ipynb` to see or modify data cleaning steps.

2. **Training:**
   - Open and run **all cells** in `hausa_train.ipynb`.
   - This will train several models, select the best, and save results to `models/hausa_sentiment/`.

3. **Evaluation:**
   - Open and run `hausa_eval.ipynb` to evaluate the saved models on the test set.
   - Results and metrics will be saved in `reports/hausa_sentiment/`.

## Sample Data

- All necessary cleaned splits are in `data/`:
  - `afrisenti_twitter_hausa_train_clean.csv`
  - `afrisenti_twitter_hausa_validation_clean.csv`
  - `afrisenti_twitter_hausa_test_clean.csv`

## Troubleshooting

- If you encounter missing packages, install them using pip or conda as above.
- For any issues, check the printed output in the notebooks for guidance.

## References

- [HausaNLP/AfriSenti-Twitter Dataset](https://huggingface.co/datasets/HausaNLP/AfriSenti-Twitter)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [HausaBERTa Model](https://huggingface.co/mangaphd/HausaBERTa)

---
