# Python Mini AI 🤖🧠

**Python Mini AI** ist eine vollständig lokale, autonome Python-KI-Anwendung mit einem **eigenen in NumPy implementierten neuronalen Netzwerk**, NLP-Textverarbeitung, regelbasiertem Entscheidungs-Engine, Chat-Gedächtnis (Kurzzeit & SQLite Langzeit) sowie einem aktiven Lern- und Feedbacksystem.

---

## 🌟 Hauptmerkmale

* **Eigenes Neuronales Netz (NumPy MLP):** Eigenständiges Multi-Layer Perceptron mit Forward Propagation, Softmax Output, Cross-Entropy Loss, Backpropagation, Adam-Optimizer und Gewicht-Persistenz (.npz).
* **NLP-Pipeline:** Eigener Tokenizer, Stopwort-Filterung, Bag-of-Words-Vektorisierung sowie Jaccard- & Cosine-Ähnlichkeitsberechnung.
* **Hybrid Decision Engine:** Priorisierte Antwortgenerierung mit transparente Confidence Scores (Knowledge Base → Neural Network → Short/Long-Term Memory → Fallback).
* **Lernfähigkeit & Feedback:** Speichert positives Feedback, erweitert Trainingsmuster und erlaubt direkte Interaktion zur Re-Trainierung des neuronalen Netzes.
* **Modernes GUI & CLI-Fallback:** Tkinter Dark Mode Desktop Interface mit Echtzeit-Confidence Gauge & Debug-Panel sowie einem automatischen Fallback für Terminal/Headless-Umgebungen.

---

## 📁 Projektstruktur

```text
python_mini_ai/
│
├── main.py                         # Anwendungseinstiegspunkt
├── config.py                       # Einstellungen, Pfade & Hyperparameter
│
├── ai/                             # Neuronales Netz & Entscheidungslogik
│   ├── __init__.py
│   ├── neural_network.py           # NumPy MLP (Forward, Backprop, Loss, Adam)
│   ├── trainer.py                  # Vektorisierung & Training Loop
│   ├── inference.py                # Vorhersage-Engine
│   ├── intent_classifier.py        # Intent Klassifikator
│   ├── response_generator.py       # Antwortgenerator
│   └── decision_engine.py          # Prioritäten- & Scoring-Engine
│
├── nlp/                            # Textverarbeitung & Similarity
│   ├── __init__.py
│   ├── tokenizer.py                # Regex Tokenizer (DE / EN)
│   ├── text_processor.py           # Normalisierung & Bag-of-Words
│   └── similarity.py               # Cosine & Jaccard Ähnlichkeiten
│
├── memory/                         # Chat-Gedächtnis
│   ├── __init__.py
│   ├── short_term.py               # Sliding Window Kontext-Puffer
│   ├── long_term.py                # SQLite Datenbank (Verlauf & Fakten)
│   └── memory_manager.py           # Memory Facade
│
├── knowledge/                      # Lokale Wissensdatenbank
│   ├── __init__.py
│   ├── knowledge_base.py           # Suche in lokaler JSON
│   └── knowledge.json              # Wissenseinträge
│
├── training/                       # Datasets & Lernsystem
│   ├── __init__.py
│   ├── dataset.py                  # Dataset Manager
│   ├── learning_system.py          # Feedback & Re-Training System
│   └── training_data.json          # Intent Training Patterns
│
├── ui/                             # Desktop Benutzeroberfläche
│   ├── __init__.py
│   └── chat_window.py              # Tkinter Dark Mode GUI
│
├── models/                         # Gespeicherte Gewichte
│   └── model_weights.npz
│
├── data/                           # Datenbanken
│   └── memory.db
│
└── tests/                          # Automated Pytest Suite
    ├── test_ai.py
    ├── test_nlp.py
    └── test_memory.py
```

---

## 🚀 Installation & Start

### 1. Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

### 2. Anwendung starten

```bash
python main.py
```

* **GUI-Modus:** Öffnet das moderne Desktop-Fenster, sofern ein Display verfügbar ist.
* **CLI-Modus:** Startet automatisch im Terminal, falls kein Display vorhanden ist.

---

## 🧠 Neural Network Architektur

1. **Input Layer:** Bag-of-Words Vektor der Eingabe basierend auf dem aktuellen Vokabular.
2. **Hidden Layers:**
   * Hidden Layer 1 (64 Neuronen, ReLU Aktivierung)
   * Hidden Layer 2 (32 Neuronen, ReLU Aktivierung)
3. **Output Layer:** Softmax Aktivierung mit Ausgabe von Wahrscheinlichkeiten für jede Intent-Klasse.
4. **Optimization:** Adam Optimizer mit Backpropagation und Cross-Entropy Loss.

---

## 🧪 Tests ausführen

```bash
PYTHONPATH=. pytest python_mini_ai/tests
```

---

## 💬 Beispiel-Interaktion (CLI)

```text
User > Was ist Python?
AI   > Python ist eine vielseitige, übersichtliche und weit verbreitete High-Level-Programmiersprache.
       [Confidence: 100.0% | Source: knowledge_base]

User > Hallo!
AI   > Hallo! Wie kann ich dir heute helfen?
       [Confidence: 98.7% | Source: neural_network]
```
