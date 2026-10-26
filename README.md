# Artificial Neural Network on the Breast Cancer Dataset

A from-scratch implementation of a feed-forward artificial neural network (ANN) in Python and NumPy, applied to the Breast Cancer Wisconsin dataset (569 tumours, 30 measurements, malignant or benign).

## How the ANN works

- **Structure:** input layer (30 features), two hidden layers (16 and 8 neurons, ReLU), one output neuron (sigmoid) that gives the probability of "benign".
- **Forward pass:** each layer multiplies its inputs by weights, adds a bias and applies an activation function.
- **Loss:** binary cross-entropy, which measures how far the predicted probabilities are from the true labels.
- **Backpropagation:** the chain rule is used to find how much each weight contributed to the error.
- **Training:** mini-batch gradient descent nudges every weight in the direction that lowers the loss. Data is shuffled every epoch.
- Weights start with He initialisation, and the features are standardized using the training data only.

## What this project does

- Loads the dataset and saves it as `data/breast_cancer.csv`
- Splits it 60% train / 20% validation / 20% test, keeping the class balance
- Trains the network for 100 epochs and records loss and accuracy per epoch
- Reports test accuracy, a confusion matrix and a classification report
- Compares with scikit-learn's MLPClassifier
- Saves the training curves as `results/training_curves.png`

## Results

Test accuracy is about 98% (see the output of `python run.py`).

## Setup

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Run

    python run.py

## Tests

    python -m pytest -v

The tests include a gradient check that compares backpropagation against a numerical estimate, and a check that the network can learn XOR.

## Files

- `src/ann.py` : the neural network
- `src/data.py` : data loading and standardization
- `run.py` : runs the full experiment
- `tests/test_ann.py` : pytest tests
- `data/breast_cancer.csv` : the dataset
- `results/` : output plot
