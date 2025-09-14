# Hausa Sentiment Analysis Pipeline

This project demonstrates a simple, modern workflow for sentiment analysis on Hausa text using pretrained transformer models and the Hausa Sentiments Corpus from Hugging Face.

## Features

- Uses Hugging Face `transformers` and `datasets` for easy, reproducible NLP workflows
- Fine-tunes a pretrained Hausa language model (`mangaphd/HausaBERTa`)
- Evaluates model performance (accuracy, F1)
- Predicts sentiment for new Hausa text samples

## Dataset

- **Name:** Hausa Sentiments Corpus
- **Source:** [Hugging Face Datasets](https://huggingface.co/datasets/michsethowusu/hausa-sentiments-corpus)
- **Splits:** Train, Test, Validation
- **Labels:** 0 = Negative, 1 = Neutral, 2 = Positive

## Quickstart

1. **Install dependencies** (in the notebook or terminal):
   ```bash
   pip install transformers datasets torch scikit-learn
   ```
2. **Open and run the notebook:**
   - `hausa_sentiment_pipeline.ipynb`
   - The notebook will:
     - Load the Hausa Sentiments Corpus
     - Tokenize the text using a pretrained Hausa model
     - Fine-tune the model on the training set
     - Evaluate on the test set
     - Predict sentiment for new Hausa sentences

## Model

- **Pretrained Model:** [`mangaphd/HausaBERTa`](https://huggingface.co/mangaphd/HausaBERTa)
- **Task:** Sequence classification (sentiment analysis)
- **Metrics:** Accuracy, F1 (weighted)

## Example Prediction

```python
sample_texts = [
    "Ina son wannan fim din sosai!",  # I love this movie!
    "Ban ji dadin wannan ba.",         # I didn't like this.
]
predictions = predict_sentiment(sample_texts)
print('Predictions:', predictions)
```

## Results

- Typical accuracy and F1 scores will be printed after training.
- You can easily swap in other Hausa models from Hugging Face if desired.

## Reproducibility

- All steps are in the notebook for transparency and easy modification.
- No complex dependencies or custom modules required.

## References

- [Hausa Sentiments Corpus on Hugging Face](https://huggingface.co/datasets/michsethowusu/hausa-sentiments-corpus)
- [HausaBERTa Model](https://huggingface.co/mangaphd/HausaBERTa)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)

---

For questions or improvements, please open an issue or pull request.
