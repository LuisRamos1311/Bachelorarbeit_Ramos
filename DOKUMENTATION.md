# Dokumentation: Evaluation moderner Vorhersagemethoden für Kryptowährungen

## Überblick

Diese Bachelorarbeit implementiert ein umfassendes Framework zur Evaluation verschiedener Machine Learning-Modelle für die Vorhersage von Kryptowährungspreisen. Das Hauptziel ist es, zu untersuchen, wie unterschiedliche Kombinationen von Eingabevariablen (Features) die Vorhersagegenauigkeit verschiedener Modelle beeinflussen.

## Wissenschaftliche Fragestellung

**Hauptforschungsfrage:** Welche Kombination aus Vorhersagemethode und Inputvariablen liefert die beste Performance bei der Prognose von Kryptowährungspreisen?

**Teilfragen:**
1. Wie unterscheiden sich moderne neuronale Netze (LSTM, GRU, RNN) und klassische Zeitreihenmodelle (ARIMA) in ihrer Performance?
2. Welchen Einfluss haben technische Indikatoren auf die Vorhersagegenauigkeit?
3. Gibt es ein optimales Set von Features für verschiedene Modelltypen?

## Methodologie

### 1. Datenerhebung
- **Quelle:** Yahoo Finance API über yfinance
- **Standardkonfiguration:** Bitcoin (BTC-USD), 2 Jahre historische Daten
- **Datenaufteilung:** 80% Training, 20% Test

### 2. Feature Engineering

Das Framework berechnet automatisch folgende technische Indikatoren:

**Gleitende Durchschnitte:**
- MA_7, MA_14, MA_30 (Simple Moving Averages)
- EMA_12, EMA_26 (Exponential Moving Averages)

**Momentum-Indikatoren:**
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)

**Volatilität:**
- Bollinger Bands (Upper, Lower, Middle)
- Rolling Volatility

**Preisveränderungen:**
- Tägliche prozentuale Veränderung
- 7-Tage prozentuale Veränderung

### 3. Feature-Sets

Sechs vordefinierte Feature-Kombinationen werden getestet:

1. **price_only:** Nur Schlusskurs (Baseline)
2. **ohlcv:** Open, High, Low, Close, Volume
3. **price_ma:** Preis mit gleitenden Durchschnitten
4. **technical_basic:** Grundlegende technische Indikatoren
5. **technical_full:** Erweiterte technische Indikatoren
6. **all_features:** Alle verfügbaren Features

### 4. Modelle

#### LSTM (Long Short-Term Memory)
- Spezialisiert auf Langzeit-Abhängigkeiten
- 2 LSTM-Schichten mit Dropout
- Gut geeignet für komplexe zeitliche Muster

#### GRU (Gated Recurrent Unit)
- Vereinfachte LSTM-Variante
- Schnelleres Training
- Ähnliche Performance bei weniger Parametern

#### Simple RNN (Recurrent Neural Network)
- Basis-Architektur für Sequenzen
- Referenzmodell für Vergleich
- Anfällig für Vanishing Gradient

#### ARIMA (AutoRegressive Integrated Moving Average)
- Klassisches statistisches Modell
- Benchmark für neuronale Netze
- Interpretierbare Parameter

### 5. Evaluationsmetriken

**Primäre Metriken:**
- **RMSE** (Root Mean Squared Error): Standardabweichung der Residuen
- **MAE** (Mean Absolute Error): Durchschnittlicher absoluter Fehler
- **MAPE** (Mean Absolute Percentage Error): Prozentualer Fehler

**Sekundäre Metriken:**
- **R²** (Bestimmtheitsmaß): Anteil erklärter Varianz
- **Directional Accuracy:** Prozentsatz korrekt vorhergesagter Richtungen

## Implementierung

### Architektur

```
src/
├── data/
│   └── data_loader.py       # Datenladung und Feature Engineering
├── models/
│   └── predictors.py        # Modellimplementierungen
├── evaluation/
│   └── metrics.py           # Evaluationsframework
└── utils/
    └── visualization.py     # Visualisierungen
```

### Arbeitsablauf

1. **Daten laden:** `CryptoDataLoader` holt Daten von Yahoo Finance
2. **Features berechnen:** Technische Indikatoren werden automatisch hinzugefügt
3. **Sequenzen erstellen:** Zeitreihen werden in Sequenzen fester Länge umgewandelt
4. **Training:** Jedes Modell wird mit jedem Feature-Set trainiert
5. **Evaluation:** Metriken werden berechnet und gespeichert
6. **Visualisierung:** Ergebnisse werden grafisch dargestellt

### Verwendung

**Vollständige Evaluation:**
```bash
python main.py
```

**Schnelles Beispiel:**
```bash
python example.py
```

**Tests ausführen:**
```bash
python test_framework.py
```

## Ergebnisinterpretation

### Output-Dateien

1. **evaluation_results.csv:** Tabellarische Ergebnisse aller Experimente
2. **plots/comparison_*.png:** Vergleichsdiagramme verschiedener Metriken
3. **plots/heatmap_*.png:** Heatmaps der Modellperformance
4. **plots/features_*.png:** Feature-Wichtigkeit pro Modell

### Interpretationsleitfaden

**Bei RMSE/MAE/MAPE:** Niedrigere Werte = bessere Performance
**Bei R²:** Höhere Werte = bessere Performance (max. 1.0)
**Bei Directional Accuracy:** Höhere Werte = bessere Richtungsvorhersage

### Typische Erkenntnisse

Basierend auf ähnlichen Studien können folgende Muster erwartet werden:

1. **Feature-Komplexität:** Mehr Features führen nicht immer zu besserer Performance
2. **Modell-Eignung:** Deep Learning übertrifft oft ARIMA, besonders bei komplexen Mustern
3. **Overfitting-Risiko:** Zu viele Features können zu Überanpassung führen
4. **Trade-off:** Balance zwischen Genauigkeit und Rechenzeit

## Anpassung und Erweiterung

### Andere Kryptowährungen

```python
# In main.py:
SYMBOL = "ETH-USD"  # Ethereum
SYMBOL = "BNB-USD"  # Binance Coin
SYMBOL = "ADA-USD"  # Cardano
```

### Hyperparameter-Tuning

```python
# Beispiel für LSTM-Anpassung:
models = {
    'LSTM': LSTMPredictor(
        units=100,        # Erhöhe Neuronen
        dropout=0.3,      # Erhöhe Dropout
        epochs=100,       # Mehr Trainingsepochs
        batch_size=64     # Größere Batches
    )
}
```

### Eigene Feature-Sets

In `src/data/data_loader.py`, Methode `get_feature_sets()`:

```python
feature_sets = {
    'my_custom_set': ['Close', 'Volume', 'RSI', 'MACD'],
    # ... weitere Sets
}
```

### Neue Modelle hinzufügen

1. Erstelle neue Klasse, die von `BasePredictor` erbt
2. Implementiere `fit()` und `predict()` Methoden
3. Füge zu `models` Dictionary in `main.py` hinzu

## Technische Anforderungen

### Hardware-Empfehlungen

- **Minimum:** 8 GB RAM, CPU mit 4 Kernen
- **Empfohlen:** 16 GB RAM, GPU (für schnelleres Training)
- **Speicherplatz:** ~2 GB für Dependencies, ~100 MB für Daten

### Software-Anforderungen

- Python 3.8+
- Internet-Verbindung (für Datendownload)
- Alle in requirements.txt aufgelisteten Pakete

### Ausführungszeit

- **Test-Script:** ~30 Sekunden
- **Beispiel-Script:** ~5-10 Minuten
- **Vollständige Evaluation:** ~30-60 Minuten (abhängig von Hardware)

## Limitierungen

1. **Datenqualität:** Abhängig von Yahoo Finance Verfügbarkeit
2. **Marktvolatilität:** Modelle können bei extremen Ereignissen versagen
3. **Stationarität:** Krypto-Märkte sind nicht-stationär
4. **Overfitting:** Besonders bei kleinen Datensätzen
5. **Keine Echtzeit-Daten:** Historische Daten nur

## Weiterentwicklung

Mögliche Erweiterungen:

1. **Sentiment-Analyse:** Twitter/Reddit-Daten integrieren
2. **Ensemble-Methoden:** Kombination mehrerer Modelle
3. **Transfer Learning:** Vortrainierte Modelle nutzen
4. **Multi-Step Forecasting:** Mehrere Zeitschritte vorhersagen
5. **Online Learning:** Kontinuierliche Modellanpassung

## Zitation und Referenzen

### Verwendete Technologien

- **TensorFlow/Keras:** Deep Learning Framework
- **Prophet:** Facebook's Zeitreihen-Tool (Option)
- **statsmodels:** ARIMA Implementation
- **yfinance:** Yahoo Finance API Wrapper
- **scikit-learn:** ML-Utilities und Metriken

### Empfohlene Literatur

1. Hochreiter & Schmidhuber (1997): "Long Short-Term Memory"
2. Cho et al. (2014): "Learning Phrase Representations using RNN Encoder-Decoder"
3. Box & Jenkins (1970): "Time Series Analysis: Forecasting and Control"

## Support und Kontakt

Bei Fragen zur Implementierung:
1. Überprüfe README.md für Basis-Informationen
2. Führe test_framework.py aus, um Installation zu validieren
3. Starte mit example.py für schnelle Tests

## Lizenz und Verwendung

Dieses Projekt ist Teil einer akademischen Bachelorarbeit und dient Bildungszwecken.
