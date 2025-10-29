# Quick Start Guide / Schnellstart-Anleitung

## English

### Installation

1. Clone the repository:
```bash
git clone https://github.com/LuisRamos1311/Bachelorarbeit_Ramos.git
cd Bachelorarbeit_Ramos
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Framework

#### Option 1: Quick Test (30 seconds)
Verify everything works:
```bash
python test_framework.py
```

#### Option 2: Example Run (5-10 minutes)
Quick demonstration with reduced parameters:
```bash
python example.py
```

#### Option 3: Full Evaluation (30-60 minutes)
Complete evaluation with all models and feature sets:
```bash
python main.py
```

### What You'll Get

After running the scripts, you'll have:
- **CSV file** with detailed results
- **plots/** folder with visualizations
- Console output with summary statistics

### Customization

Edit the scripts to customize:
- Cryptocurrency symbol (e.g., "ETH-USD", "ADA-USD")
- Time period (e.g., "1y", "2y", "5y")
- Model parameters (epochs, batch size, etc.)
- Feature combinations

---

## Deutsch

### Installation

1. Repository klonen:
```bash
git clone https://github.com/LuisRamos1311/Bachelorarbeit_Ramos.git
cd Bachelorarbeit_Ramos
```

2. Abhängigkeiten installieren:
```bash
pip install -r requirements.txt
```

### Framework ausführen

#### Option 1: Schnelltest (30 Sekunden)
Überprüfen, ob alles funktioniert:
```bash
python test_framework.py
```

#### Option 2: Beispiel-Durchlauf (5-10 Minuten)
Schnelle Demonstration mit reduzierten Parametern:
```bash
python example.py
```

#### Option 3: Vollständige Evaluation (30-60 Minuten)
Komplette Evaluation mit allen Modellen und Feature-Sets:
```bash
python main.py
```

### Was Sie erhalten

Nach dem Ausführen der Skripte haben Sie:
- **CSV-Datei** mit detaillierten Ergebnissen
- **plots/** Ordner mit Visualisierungen
- Konsolenausgabe mit Zusammenfassungsstatistiken

### Anpassung

Bearbeiten Sie die Skripte, um anzupassen:
- Kryptowährungs-Symbol (z.B. "ETH-USD", "ADA-USD")
- Zeitraum (z.B. "1y", "2y", "5y")
- Modellparameter (Epochen, Batch-Größe, etc.)
- Feature-Kombinationen

---

## Troubleshooting / Fehlerbehebung

### Problem: Import errors
**Solution:** Make sure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

### Problem: Data download fails
**Solution:** Check internet connection. Yahoo Finance might be temporarily unavailable.

### Problem: Out of memory
**Solution:** Reduce batch size or use shorter time period in the script.

### Problem: Slow execution
**Solution:** 
- Use GPU if available
- Reduce epochs in model configuration
- Use example.py instead of main.py

---

## System Requirements

### Minimum
- Python 3.8+
- 8 GB RAM
- 4-core CPU
- Internet connection

### Recommended
- Python 3.9+
- 16 GB RAM
- GPU (NVIDIA with CUDA)
- Fast internet connection

---

## File Structure

```
.
├── main.py              # Full evaluation
├── example.py           # Quick example
├── test_framework.py    # Tests
├── requirements.txt     # Dependencies
├── README.md           # Main documentation
├── DOKUMENTATION.md    # Detailed German docs
├── QUICKSTART.md       # This file
└── src/
    ├── data/           # Data loading
    ├── models/         # Model implementations
    ├── evaluation/     # Metrics
    └── utils/          # Visualization
```

---

## Next Steps

1. ✅ Install dependencies
2. ✅ Run test_framework.py
3. ✅ Run example.py
4. 📊 Analyze results
5. 🔧 Customize for your needs
6. 🚀 Run full evaluation

For detailed documentation, see:
- **README.md** - Overview and basic usage
- **DOKUMENTATION.md** - Comprehensive German documentation
