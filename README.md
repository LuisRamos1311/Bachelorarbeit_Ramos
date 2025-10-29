# Bachelorarbeit: Evaluation moderner Vorhersagemethoden für Kryptowährungen

## Beschreibung / Description

**Deutsch:** Evaluation moderner Vorhersagemethoden im Kontext verschiedener Inputvariablen für Kryptowährungen

**English:** Evaluation of modern prediction methods in the context of different input variables for cryptocurrencies

Bachelorarbeit Wirtschaftsinformatik

## Projekt-Übersicht / Project Overview

Dieses Projekt implementiert und vergleicht verschiedene moderne Machine Learning-Modelle zur Vorhersage von Kryptowährungspreisen. Es werden verschiedene Kombinationen von Eingabevariablen (Features) getestet, um die optimale Konfiguration zu ermitteln.

This project implements and compares various modern machine learning models for cryptocurrency price prediction. Different combinations of input variables (features) are tested to determine the optimal configuration.

### Implementierte Modelle / Implemented Models

- **LSTM** (Long Short-Term Memory)
- **GRU** (Gated Recurrent Unit)
- **Simple RNN** (Recurrent Neural Network)
- **ARIMA** (AutoRegressive Integrated Moving Average)

### Feature-Sets / Feature Sets

1. **price_only**: Nur Schlusskurs / Close price only
2. **ohlcv**: Open, High, Low, Close, Volume
3. **price_ma**: Preis + Moving Averages
4. **technical_basic**: Grundlegende technische Indikatoren
5. **technical_full**: Vollständige technische Indikatoren (MA, EMA, MACD, RSI, etc.)
6. **all_features**: Alle verfügbaren Features

### Evaluationsmetriken / Evaluation Metrics

- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coefficient of Determination)
- Directional Accuracy

## Installation

### Voraussetzungen / Prerequisites

- Python 3.8 oder höher / Python 3.8 or higher
- pip package manager

### Abhängigkeiten installieren / Install Dependencies

```bash
pip install -r requirements.txt
```

## Verwendung / Usage

### Hauptskript ausführen / Run Main Script

```bash
python main.py
```

Das Skript führt folgende Schritte aus:
1. Lädt Kryptowährungsdaten (Standard: Bitcoin)
2. Berechnet technische Indikatoren
3. Trainiert alle Modelle mit allen Feature-Sets
4. Evaluiert die Modelle
5. Erstellt Visualisierungen
6. Speichert Ergebnisse in CSV

The script performs the following steps:
1. Loads cryptocurrency data (default: Bitcoin)
2. Calculates technical indicators
3. Trains all models with all feature sets
4. Evaluates the models
5. Creates visualizations
6. Saves results to CSV

### Ausgabe / Output

- `evaluation_results.csv`: Detaillierte Ergebnisse aller Experimente
- `plots/`: Ordner mit allen Visualisierungen
  - Vergleichsdiagramme für verschiedene Metriken
  - Heatmaps der Modellperformance
  - Feature-Wichtigkeit für jedes Modell

## Projektstruktur / Project Structure

```
.
├── src/
│   ├── data/
│   │   └── data_loader.py      # Datenladung und -vorbereitung
│   ├── models/
│   │   └── predictors.py       # Modellimplementierungen
│   ├── evaluation/
│   │   └── metrics.py          # Evaluationsmetriken
│   └── utils/
│       └── visualization.py    # Visualisierungsfunktionen
├── main.py                     # Hauptskript
├── requirements.txt            # Python-Abhängigkeiten
└── README.md                   # Diese Datei

```

## Anpassung / Customization

### Andere Kryptowährung verwenden / Use Different Cryptocurrency

Bearbeiten Sie `main.py` und ändern Sie:
```python
SYMBOL = "ETH-USD"  # für Ethereum
# oder / or
SYMBOL = "ADA-USD"  # für Cardano
```

### Modellparameter anpassen / Adjust Model Parameters

Bearbeiten Sie die Modellinitialisierung in `main.py`:
```python
models = {
    'LSTM': LSTMPredictor(units=100, dropout=0.3, epochs=50, batch_size=64),
    # ... weitere Modelle
}
```

### Eigene Feature-Sets hinzufügen / Add Custom Feature Sets

Bearbeiten Sie `src/data/data_loader.py`, Methode `get_feature_sets()`.

## Ergebnisse / Results

Nach der Ausführung werden detaillierte Ergebnisse auf der Konsole angezeigt und in Dateien gespeichert. Die Visualisierungen helfen dabei, die Leistung der verschiedenen Modell-Feature-Kombinationen zu vergleichen.

After execution, detailed results are displayed on the console and saved to files. The visualizations help compare the performance of different model-feature combinations.

## Lizenz / License

Dieses Projekt ist Teil einer Bachelorarbeit an der Hochschule.

This project is part of a Bachelor's thesis at the university.

## Autor / Author

Luis Ramos - Bachelorarbeit Wirtschaftsinformatik
