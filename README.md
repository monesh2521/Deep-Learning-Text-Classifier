# Task 5 — Deep Learning / NLP Text Classifier with PyTorch

## Project contents
- `data/sentiment_dataset.csv` — balanced educational sentiment dataset (120 positive and 120 negative examples)
- `src/train.py` — Logistic Regression baseline and PyTorch neural network, evaluated on the same stratified test split
- `requirements.txt` — required Python packages
- `outputs/` — saved model files and evaluation metrics after execution

## Labels
- Negative = 0
- Positive = 1

## Run in VS Code (Windows)
1. Extract the ZIP file.
2. Open the extracted `Task5_Corrected_Complete` folder in VS Code.
3. Select your Python interpreter.
4. Open Terminal → New Terminal.
5. Install packages:
   `python -m pip install -r requirements.txt`
6. Run the project:
   `python src/train.py`

## Output
The program prints dataset balance, Logistic Regression accuracy/report/confusion matrix, training progress, and PyTorch accuracy/report/confusion matrix. It saves model files and `metrics.txt` in `outputs/`.

## Important
The dataset is a small, manually created educational dataset. It is intended to demonstrate an NLP workflow, not to claim production-level performance. Always use the actual measured test scores in your assignment report.
