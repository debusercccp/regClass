# Guida a libNN

Una libreria per il Deep Learning scritta da zero in Python e NumPy.

---

## Installazione (Consigliata)

Per rendere la libreria importabile da qualsiasi cartella nel tuo sistema senza impazzire con il `PYTHONPATH`, usa il file `pyproject.toml` incluso.

Assicurati di avere il file `pyproject.toml` nella cartella radice (`~/libNN/`):

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "libNN"
version = "0.1.0"
dependencies = ["numpy", "matplotlib"]

[tool.setuptools]
packages = ["libNN"]
package-dir = {"libNN" = "."}
```

Dalla radice della libreria, installa in modalità editable:

```bash
pip install -e .
```

Questo ti permette di modificare il codice in `libNN/` e vedere i cambiamenti riflettersi immediatamente nei tuoi script senza reinstallare.

---

## Struttura della Libreria

- **`network.py`**: Contiene `NeuralNetwork` (per assemblare i layer) e `Trainer` (per il ciclo di addestramento).
- **`losses.py`**: Contiene le funzioni di Loss (`MeanSquaredError`, `BinaryCrossEntropy`, `DiceLoss`, `CategoricalCrossEntropy`).
- **`layers.py`**: I layer standard: `Dense` (neuroni completamente connessi) e `BatchNorm` (normalizzazione per batch).
- **`activation.py`**: Le funzioni di attivazione (`ReLU`, `LeakyReLU`, `Tanh`, `Sigmoid`, `Linear`, `Softmax`) che danno "intelligenza" e non-linearità alla rete.
- **`optimizers.py`**: I motori di ricerca del minimo (`SGD`, `SGDMomentum`, `Adam`, tutti con `weight_decay` opzionale) che aggiornano i pesi.
- **`operations.py`**: Operazioni speciali come il `Dropout` (per la regolarizzazione).

---

## Costruire una Rete Standard (Feedforward)

Usa la classe `NeuralNetwork` combinata con i layer `Dense` quando hai dati tabulari (es. file CSV) o problemi matematici classici (come la regressione).

```python
from libNN import NeuralNetwork, Dense, ReLU, Dropout, Linear
from libNN.losses import MeanSquaredError

layers = [
    Dense(neurons=64, activation=ReLU()),
    Dropout(keep_prob=0.8),
    Dense(neurons=32, activation=ReLU()),
    Dense(neurons=1, activation=Linear())
]

model = NeuralNetwork(layers=layers, loss=MeanSquaredError())
```

---

## Layer Disponibili

| Layer | Descrizione |
|---|---|
| `Dense(neurons, activation)` | Layer completamente connesso. Inizializzazione He per ReLU/LeakyReLU, scelta automatica del numero di input dalla forma del primo batch. |
| `BatchNorm(num_features, momentum=0.1, eps=1e-5)` | Normalizzazione per batch con parametri learnable γ (scala) e β (shift). |
| `Dropout(keep_prob=0.8)` | Regolarizzazione (inverted dropout). Trasparente in inferenza. |

### `BatchNorm`

Durante il training normalizza rispetto a media/varianza del batch corrente
(varianza di popolazione, divisa per `B`, come PyTorch con `affine=True`) e
aggiorna le statistiche running con una media mobile esponenziale. In inferenza
usa le statistiche running accumulate. Va inserito **prima** dell'attivazione
non lineare del layer successivo.

```python
from libNN import NeuralNetwork, Dense, BatchNorm, Dropout, ReLU, Linear
from libNN.losses import MeanSquaredError

layers = [
    Dense(neurons=64, activation=ReLU()),
    BatchNorm(64),
    Dropout(keep_prob=0.8),
    Dense(neurons=1, activation=Linear()),
]
model = NeuralNetwork(layers=layers, loss=MeanSquaredError())
```

---

## Scegliere la Loss Giusta

La funzione di Loss è l'obiettivo della tua rete. Se scegli quella sbagliata, la rete non imparerà nulla.

| Loss | Usala per | Ultimo layer |
|---|---|---|
| `MeanSquaredError()` | **Regressione** — predire un numero continuo (es. temperatura, prezzo di una casa) | `Linear()` |
| `BinaryCrossEntropy()` | **Classificazione Binaria** — due classi (es. cane/gatto, sfondo/primo piano) | `Sigmoid()` |
| `DiceLoss()` | **Segmentazione Sbilanciata** — trovare piccole regioni in immagini grandi | `Sigmoid()` |
| `CategoricalCrossEntropy()` | **Classificazione Multi-classe** — tre o più classi mutualmente esclusive | `Softmax()` |

> `CategoricalCrossEntropy` richiede che le etichette `y` siano in formato **one-hot encoded**
> (es. classe 2 su 3 classi → `[0, 0, 1]`).
> `Softmax` e `CategoricalCrossEntropy` sono progettate per lavorare **in coppia**: usarle
> separatamente con altre combinazioni può dare risultati inaspettati.

---

## Funzioni di Attivazione

| Attivazione | Quando usarla |
|---|---|
| `ReLU()` | Layer nascosti — scelta predefinita, veloce ed efficace |
| `LeakyReLU(alpha=0.2)` | Layer nascosti — evita il problema dei "neuroni morti" di ReLU |
| `Tanh()` | Layer nascosti — output zero-centrato in (−1, 1) |
| `Sigmoid()` | Output per classificazione binaria o segmentazione |
| `Softmax()` | Output per classificazione multi-classe — converte i logit in probabilità (somma = 1) |
| `Linear()` | Output per regressione — lascia passare il valore senza modifiche |

---

## Addestrare il Modello (Il Trainer)

Indipendentemente da quale rete o loss hai scelto, l'addestramento si fa sempre allo stesso modo grazie alla classe `Trainer`.

```python
from libNN.optimizers import SGDMomentum
from libNN.network import Trainer

optimizer = SGDMomentum(lr=0.01, momentum=0.9)
trainer = Trainer(net=model, optim=optimizer)

history = trainer.fit(
    X_train, y_train,   # Dati per imparare
    X_test,  y_test,    # Dati di validazione (opzionali: passa None per disattivarli)
    epochs=100,         # Quante volte guardare tutti i dati
    eval_every=10,      # Ogni quante epoche valutare la Loss di validazione
    batch_size=32,      # Quanti dati processare alla volta
    patience=5          # Epoche di valutazione senza miglioramento prima di fermarsi
)

# fit() restituisce uno storico dell'addestramento:
print(history['train_loss'])     # lista: loss media per epoca
print(history['val_loss'])       # lista: validation loss per valutazione (vuota se val=None)
print(history['stopped_epoch'])  # epoca di early stopping, o None
```

### Ottimizzatori

| Ottimizzatore | Quando usarlo |
|---|---|
| `SGD(lr=0.01)` | Baseline semplice |
| `SGDMomentum(lr=0.01, momentum=0.9)` | Scelta predefinita — convergenza più stabile |
| `Adam(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)` | Reti profonde o gradienti sparsi |

Tutti gli ottimizzatori accettano `weight_decay` (regolarizzazione L2). Il decay
si applica solo alle matrici dei pesi, **mai** ai bias né ai parametri di `BatchNorm`,
in linea con la pratica standard.

```python
from libNN.optimizers import Adam, SGDMomentum

Adam(lr=0.001, weight_decay=1e-4)
SGDMomentum(lr=0.01, momentum=0.9, weight_decay=1e-3)
```

### Parametri di `fit()`

| Parametro | Default | Descrizione |
|---|---|---|
| `epochs` | `100` | Numero massimo di epoche di addestramento |
| `eval_every` | `10` | Ogni quante epoche valutare sul validation set |
| `batch_size` | `32` | Dimensione dei mini-batch |
| `seed` | `1` | Seed per la riproducibilità dello shuffle |
| `restart` | `True` | Se `True`, reinizializza i layer prima di partire |
| `patience` | `5` | Numero di valutazioni consecutive senza miglioramento prima dell'early stopping |

### Early Stopping

Il `Trainer` monitora la validation loss ad ogni valutazione. Se la loss non migliora per `patience` valutazioni consecutive, l'addestramento si interrompe automaticamente e il modello viene riportato al miglior checkpoint registrato.

```
Epoch 10: Validation Loss = 0.0842 ✓
Epoch 20: Validation Loss = 0.0761 ✓
Epoch 30: Validation Loss = 0.0798 (no miglioramento, patience 1/5)
Epoch 40: Validation Loss = 0.0823 (no miglioramento, patience 2/5)
...
Early stopping attivato all'epoch 80. Ripristino miglior modello (loss=0.0761).
```

> Con `eval_every=10` e `patience=5`, il training si ferma al massimo dopo 50 epoche consecutive senza miglioramento.

---

## Salvataggio e Caricamento

Il salvataggio preserva l'intera struttura dei layer, i pesi, i bias e lo stato dell'addestramento tramite `pickle`.

```python
# Salva il modello corrente
model.save_model("mio_modello.pkl")

# Carica un modello esistente
model = NeuralNetwork.load_model("mio_modello.pkl")
```

### Auto-Checkpoint (Best Model)

`check_and_save()` salva il modello **solo se** la loss corrente è inferiore al minimo storico. Utile da chiamare manualmente nel proprio loop di training.

```python
# Salva solo se è il miglior modello finora
model.check_and_save(current_loss=val_loss, filename="best_model.pkl")
```

---

## Esempi per Tipo di Task

### Regressione

```python
from libNN import NeuralNetwork, Dense, ReLU, Linear
from libNN.losses import MeanSquaredError

layers = [
    Dense(neurons=64, activation=ReLU()),
    Dense(neurons=32, activation=ReLU()),
    Dense(neurons=1,  activation=Linear()),
]
model = NeuralNetwork(layers=layers, loss=MeanSquaredError())
```

### Classificazione Binaria

```python
from libNN import NeuralNetwork, Dense, ReLU, Sigmoid
from libNN.losses import BinaryCrossEntropy

layers = [
    Dense(neurons=64, activation=ReLU()),
    Dense(neurons=32, activation=ReLU()),
    Dense(neurons=1,  activation=Sigmoid()),
]
model = NeuralNetwork(layers=layers, loss=BinaryCrossEntropy())
```

### Classificazione Multi-classe

```python
from libNN import NeuralNetwork, Dense, ReLU, Softmax
from libNN.losses import CategoricalCrossEntropy

N_CLASSES = 3

layers = [
    Dense(neurons=64, activation=ReLU()),
    Dense(neurons=32, activation=ReLU()),
    Dense(neurons=N_CLASSES, activation=Softmax()),
]
model = NeuralNetwork(layers=layers, loss=CategoricalCrossEntropy())

# y deve essere one-hot encoded:
# classe 0 → [1, 0, 0]
# classe 1 → [0, 1, 0]
# classe 2 → [0, 0, 1]
```

### Predizione e Valutazione

La rete espone metodi che disattivano automaticamente Dropout e BatchNorm
(modalità inferenza), senza dover toccare manualmente il flag `model.train`:

```python
y_probs   = model.predict(X_test)          # output grezzo (probabilità o logit)
y_pred    = model.predict_classes(X_test)  # classi discrete (soglia 0.5 o argmax)
test_loss = model.evaluate(X_test, y_test) # loss su un batch, senza aggiornare i pesi
```

| Metodo | Descrizione |
|---|---|
| `model.forward(x)` | Passaggio in avanti (rispetta `model.train`) |
| `model.predict(x)` | Output grezzo in modalità inferenza |
| `model.predict_classes(x)` | Classi discrete: soglia 0.5 (binario) o argmax (multi-classe) |
| `model.evaluate(x, y)` | Loss su un batch in inferenza |

---

## Utilità (`utils.py`)

```python
from libNN.utils import (
    compute_accuracy, compute_f1_score, confusion_matrix,
    mae, rmse, r2_score,
    normalize, normalize_minmax, normalize_data,
    to_one_hot, train_test_split,
)
```

| Funzione | Descrizione |
|---|---|
| `compute_accuracy(y_pred, y_true)` | Accuratezza per classificazione |
| `compute_f1_score(y_pred, y_true)` | F1-score per classificazione |
| `confusion_matrix(y_true, y_pred)` | Matrice di confusione `n_classi × n_classi` |
| `mae(y_true, y_pred)` | Mean Absolute Error per regressione |
| `rmse(y_true, y_pred)` | Root Mean Squared Error per regressione |
| `r2_score(y_true, y_pred)` | Coefficiente di determinazione R² |
| `normalize(X)` | Standardizzazione z-score per colonna: `(x − μ) / σ` |
| `normalize_minmax(X)` | Min-max per colonna: `(x − min) / (max − min)` |
| `normalize_data(X)` | Min-max globale su tutto l'array (es. immagini) |
| `to_one_hot(labels, num_classes)` | Converte etichette intere in one-hot |
| `train_test_split(*arrays, test_size, shuffle, seed)` | Divide uno o più array in train e test set |

### `train_test_split`

Replica il comportamento di `sklearn.model_selection.train_test_split`, senza dipendenze esterne.

```python
from libNN.utils import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)
```

| Parametro | Default | Descrizione |
|---|---|---|
| `*arrays` | — | Uno o più array con lo stesso numero di campioni |
| `test_size` | `0.2` | Frazione del dataset da usare come test (es. `0.2` → 20%) |
| `shuffle` | `True` | Se `True`, mescola i dati prima di dividere |
| `seed` | `None` | Seed per la riproducibilità (equivalente a `random_state` di sklearn) |

Supporta qualsiasi numero di array in ingresso:

```python
# Con un solo array
X_train, X_test = train_test_split(X, test_size=0.3)

# Con più array
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, test_size=0.2, seed=0)
```
