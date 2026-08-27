# Python Mini AI & Python Mini AI Pro 🤖🧠

**Python Mini AI** ist ein hochperformantes, vollständig lokales KI-System mit eigenem **NumPy Deep Neural Network**, NLP-Textverarbeitung, RAG VectorStore, Code Sandbox, schnelle Mathe-Engine („Rechenprofi“) sowie Kurz- und Langzeitgedächtnis (SQLite & JSON).

---

## 🚀 Key Features

* ⚡ **Ultra-Schnelle Inference & Sub-Millisekunden Mathe-Engine ("Rechenprofi"):** Löst mathematische Aufgaben (`125 * 8`, `sqrt(144)`, `2^10`) sofort mit 100% Confidence.
* 🧠 **Eigenes Deep Neural Network (NumPy MLP):** Multi-Layer Perceptron mit Forward/Backpropagation, LeakyReLU, Softmax, Cross-Entropy Loss, Adam-Optimizer und Gewicht-Persistenz (`.npz`).
* 🔍 **RAG Vector Store & Context Engine:** Lokaler Vektorspeicher mit TF-IDF & Document Chunking für Dokumentensuche.
* 💻 **Code Execution Sandbox:** Sichere Ausführung von Python-Code-Snippets in isolierter Umgebung.
* 💾 **Dual-Memory System:** Short-Term Sliding-Window Gedächtnis + Long-Term SQLite Datenbank.
* 🖥️ **Dark Mode Desktop GUI & Instant CLI Fallback:** Tkinter Dark Mode Chat-Interface mit Confidence-Anzeige und automatischem Terminal-Modus für Headless-Systeme.

---

## 📥 Detaillierte Installationsanleitung

### 1. Systemvoraussetzungen
* **Python:** Version 3.10, 3.11 oder 3.12+
* **Betriebssysteme:** Linux (Ubuntu/Debian/Fedora/Arch), macOS, Windows 10/11
* **Grafische Benutzeroberfläche (Optional):** Tkinter (auf Linux: `sudo apt install python3-tk`)

---

### 2. Repository Klonen / Navigieren

```bash
cd /pfad/zu/deinem/projekt
```

---

### 3. Virtuelle Umgebung Erstellen & Aktivieren

#### Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows (PowerShell / CMD):
```powershell
python -m venv venv
.\venv\Scripts\activate
```

---

### 4. Abhängigkeiten Installieren

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🎮 Startanleitungen

### 🔵 Python Mini AI (Standard Edition)

```bash
python main.py
```
* Startet das **Tkinter Dark Mode Interface** (sofern ein Display vorhanden ist) oder wechselt nahtlos in den **CLI Terminal Chat**.

---

### 🟣 Python Mini AI Pro (Professional Edition)

```bash
PYTHONPATH=. python3 python_mini_ai_pro/main.py
```
* Bietet erweiterte Befehle im Chat:
  * `run <code>` – Führt Code in der Sandbox aus.
  * `search <query>` – Durchsucht den RAG Vector Store.

---

## 🧪 Tests Ausführen

Um die gesamte Testsuite (15 Tests) für alle KI-Komponenten auszuführen:

```bash
PYTHONPATH=. pytest python_mini_ai/tests python_mini_ai_pro/tests
```

---

## 🎥 Demos & Interaktive Beispiele im Chat

Hier siehst du typische Interaktionen der lokalen KI:

### 1. Mathe-Engine ("Rechenprofi")
```text
User > Was ist 125 * 8?
AI   > Das Ergebnis der Rechnung '125 * 8' ist: 1000
       [Confidence: 100.0% | Source: math_engine]
```

### 2. Wissen zur Erde & Universum
```text
User > Wie groß ist das Universum?
AI   > Das beobachtbare Universum hat einen Durchmesser von schätzungsweise 93 Milliarden Lichtjahren und enthält über 2 Billionen Galaxien.
       [Confidence: 100.0% | Source: knowledge_base]
```

### 3. RAG VectorStore Suche (Pro Edition)
```text
PRO-AI User > search Python Pro
[RAG VECTOR RETRIEVAL]
1. [Python Pro Guide] Score: 0.447
   Chunk: Python Pro unterstützt RAG Vektorspeicher, Code Ausführung und neuronale Netze.
```

### 4. Code Execution Sandbox (Pro Edition)
```text
PRO-AI User > run print("Hallo aus der Sandbox!")
[SANDBOX EXECUTION]
Status: SUCCESS
Output:
Hallo aus der Sandbox!
```
