import numpy as np
from numpy import ndarray

def assert_same_shape(array: ndarray, array_grad: ndarray):
    '''
    Assert that two arrays have the same shape.
    Used throughout the library to catch gradient shape mismatches.
    
    Args:
        array: First array
        array_grad: Second array
        
    Raises:
        AssertionError: If shapes don't match
    '''
    assert array.shape == array_grad.shape, \
        f"Shapes must match: {array.shape} vs {array_grad.shape}"

def permute_data(X: ndarray, y: ndarray):
    '''
    Randomly shuffle data (useful for mini-batch SGD).
    Shuffles features and targets with the same permutation.
    
    Args:
        X: Features array, shape (n_samples, n_features)
        y: Target array, shape (n_samples, ...)
        
    Returns:
        Tuple of (X_shuffled, y_shuffled) with same permutation applied
    '''
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]

def to_one_hot(labels: ndarray, num_classes: int) -> ndarray:
    '''
    Convert class labels to one-hot encoded format.
    Used for multi-class classification with CategoricalCrossEntropy.
    
    Args:
        labels: Array of class indices, shape (n_samples,) with values in [0, num_classes)
        num_classes: Total number of classes
        
    Returns:
        One-hot encoded array, shape (n_samples, num_classes)
        
    Example:
        >>> labels = np.array([0, 2, 1, 0])
        >>> one_hot = to_one_hot(labels, num_classes=3)
        >>> one_hot
        array([[1., 0., 0.],
               [0., 0., 1.],
               [0., 1., 0.],
               [1., 0., 0.]])
    '''
    labels = labels.astype(int).flatten()
    one_hot = np.zeros((labels.shape[0], num_classes))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot

def compute_accuracy(predictions: ndarray, target: ndarray) -> float:
    '''
    Compute classification accuracy.
    Handles both one-hot encoded and integer label targets.
    
    Args:
        predictions: Model predictions, shape (n_samples, n_classes) (e.g., from Softmax)
        target: Ground truth, either one-hot (n_samples, n_classes) or indices (n_samples,)
        
    Returns:
        Accuracy as a float in [0, 1]
        
    Example:
        >>> preds = np.array([[0.1, 0.9], [0.8, 0.2]])  # 2 samples, 2 classes
        >>> targets = np.array([1, 0])  # class 1, class 0
        >>> compute_accuracy(preds, targets)
        1.0
    '''
    if predictions.shape != target.shape:
        # Handle case where target is indices, not one-hot
        pred_labels = np.argmax(predictions, axis=1)
        return np.mean(pred_labels == target.flatten())
    
    # Both are one-hot or similar shape
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(target, axis=1)
    return np.mean(pred_labels == true_labels)

def mae(y_true: ndarray, y_pred: ndarray) -> float:
    '''
    Mean Absolute Error (MAE) for regression.
    L = 1/N * sum(|y_true - y_pred|)
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values (same shape as y_true)
        
    Returns:
        MAE as a float (non-negative)
    '''
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true: ndarray, y_pred: ndarray) -> float:
    '''
    Root Mean Squared Error (RMSE) for regression.
    L = sqrt(1/N * sum((y_true - y_pred)^2))
    
    More sensitive to outliers than MAE.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values (same shape as y_true)
        
    Returns:
        RMSE as a float (non-negative)
    '''
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))

def normalize_data(data: ndarray) -> ndarray:
    '''
    Global min-max normalization: scale all data to [0, 1] using a single
    global min/max. Useful for image data.

    Args:
        data: Input array of any shape

    Returns:
        Normalized array with values in [0, 1], same shape as input

    Formula: x_norm = (x - min(x)) / (max(x) - min(x) + eps)
    '''
    return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-9)


def normalize(x: ndarray) -> ndarray:
    '''
    Z-score normalization per column: (x - mean) / std.
    Columns with (near-)zero standard deviation are centered but not scaled.

    Args:
        x: Input array, shape (n_samples, n_features)

    Returns:
        Standardized array (zero mean, unit variance per column).
    '''
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    safe_std = np.where(std > 1e-10, std, 1.0)
    return (x - mean) / safe_std


def normalize_minmax(x: ndarray) -> ndarray:
    '''
    Min-max normalization per column: (x - min) / (max - min).
    Columns with zero range are mapped to 0.

    Args:
        x: Input array, shape (n_samples, n_features)

    Returns:
        Array scaled to [0, 1] per column.
    '''
    col_min = x.min(axis=0, keepdims=True)
    col_max = x.max(axis=0, keepdims=True)
    rng = col_max - col_min
    safe_rng = np.where(rng > 1e-10, rng, 1.0)
    out = (x - col_min) / safe_rng
    # Columns with zero range collapse to 0.
    return np.where(rng > 1e-10, out, 0.0)


def r2_score(y_true: ndarray, y_pred: ndarray) -> float:
    '''
    Coefficient of determination R^2 for regression.
    R^2 = 1 - SS_res / SS_tot

    Args:
        y_true: Ground truth values
        y_pred: Predicted values (same shape as y_true)

    Returns:
        R^2 as a float (1.0 = perfect, 0.0 = predicting the mean).
    '''
    ss_res = np.sum(np.power(y_true - y_pred, 2))
    ss_tot = np.sum(np.power(y_true - np.mean(y_true), 2))
    if ss_tot < 1e-15:
        return 1.0
    return 1.0 - ss_res / ss_tot


def confusion_matrix(y_true: ndarray, y_pred: ndarray) -> ndarray:
    '''
    Confusion matrix of shape (n_classes, n_classes).
    cm[i][j] = number of samples with true label i predicted as j.

    Handles one-hot encoded inputs (multi-class) or single-column inputs
    (binary, thresholded at 0.5).

    Args:
        y_true: Ground truth (one-hot, indices, or binary column)
        y_pred: Predictions (probabilities/logits or labels)

    Returns:
        Integer confusion matrix as an ndarray.
    '''
    def to_labels(arr, n_classes_hint):
        arr = np.asarray(arr)
        if arr.ndim == 2 and arr.shape[1] > 1:
            return np.argmax(arr, axis=1), arr.shape[1]
        flat = arr.reshape(arr.shape[0], -1)[:, 0] if arr.ndim == 2 else arr.flatten()
        # Single column: treat as binary threshold if values are not clean indices.
        if n_classes_hint == 2:
            return (flat >= 0.5).astype(int), 2
        return flat.astype(int), n_classes_hint

    n_classes = max(
        y_pred.shape[1] if (y_pred.ndim == 2 and y_pred.shape[1] > 1) else 2,
        y_true.shape[1] if (y_true.ndim == 2 and y_true.shape[1] > 1) else 2,
    )
    pred_labels, _ = to_labels(y_pred, n_classes)
    true_labels, _ = to_labels(y_true, n_classes)

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm

def compute_f1_score(predictions: ndarray, target: ndarray) -> float:
    '''
    Compute F1-Score for binary classification.
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Use for: Imbalanced binary classification problems.
    
    Args:
        predictions: Predicted binary labels (0 or 1), or raw predictions to be thresholded
        target: Ground truth binary labels (0 or 1)
        
    Returns:
        F1-Score as a float in [0, 1]. Higher is better.
        
    Note:
        If predictions are soft (e.g., probabilities from Sigmoid), 
        threshold them first: pred_binary = (predictions > 0.5).astype(int)
        
    Example:
        >>> preds = np.array([1, 0, 1, 1, 0, 1])
        >>> true = np.array([1, 0, 1, 0, 0, 1])
        >>> compute_f1_score(preds, true)
        0.8333...
    '''
    # Flatten for safety
    preds = predictions.flatten()
    actual = target.flatten()

    # Compute True Positives, False Positives, False Negatives
    tp = np.sum((preds == 1) & (actual == 1))
    fp = np.sum((preds == 1) & (actual == 0))
    fn = np.sum((preds == 0) & (actual == 1))

    # Precision and Recall with small epsilon to avoid division by zero
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)

    # F1 Score = 2 * (P * R) / (P + R)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    return f1

def train_test_split(*arrays, test_size: float = 0.2, shuffle: bool = True, seed: int = None):
    """
    Divide uno o più array in coppie train/test.
    Replica il comportamento di sklearn.model_selection.train_test_split.

    Parametri
    ---------
    *arrays     : uno o più ndarray con lo stesso numero di campioni (asse 0)
    test_size   : frazione del dataset da usare come test (default 0.2 → 20%)
    shuffle     : se True, mescola i dati prima di dividere (default True)
    seed        : seed per la riproducibilità (default None)

    Ritorna
    -------
    Lista alternata [train_1, test_1, train_2, test_2, ...] — stesso ordine di sklearn.

    Esempi
    ------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)
    X_train, X_test = train_test_split(X, test_size=0.3)
    """
    if not arrays:
        raise ValueError("Passa almeno un array.")

    n_samples = arrays[0].shape[0]
    for i, arr in enumerate(arrays):
        if arr.shape[0] != n_samples:
            raise ValueError(
                f"Tutti gli array devono avere lo stesso numero di campioni. "
                f"arrays[0] ha {n_samples}, arrays[{i}] ha {arr.shape[0]}."
            )

    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size deve essere tra 0 e 1, ricevuto {test_size}.")

    if seed is not None:
        np.random.seed(seed)

    indices = np.random.permutation(n_samples) if shuffle else np.arange(n_samples)

    n_test  = max(1, int(np.floor(test_size * n_samples)))
    n_train = n_samples - n_test

    train_idx = indices[:n_train]
    test_idx  = indices[n_train:]

    result = []
    for arr in arrays:
        result.append(arr[train_idx])
        result.append(arr[test_idx])

    return result
