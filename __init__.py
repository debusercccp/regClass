# libNN/__init__.py

# Struttura principale
from .network import NeuralNetwork, Trainer

# Funzioni di Perdita (Loss)
from .losses import MeanSquaredError, BinaryCrossEntropy, DiceLoss, CategoricalCrossEntropy

# Layer e componenti
from .layers import Dense, BatchNorm
from .activation import ReLU, LeakyReLU, Sigmoid, Tanh, Linear, Softmax
from .operations import Dropout

# Ottimizzatori
from .optimizers import SGD, SGDMomentum, Adam

# Utility
from .utils import (
    compute_accuracy, compute_f1_score, confusion_matrix,
    mae, rmse, r2_score,
    normalize, normalize_minmax, normalize_data,
    to_one_hot, train_test_split,
)
