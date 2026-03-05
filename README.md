# 🎵 Mantra Prototype

Structural MIDI Similarity Engine for Music Analysis & IP Protection

Mantra is an interval-based music fingerprinting engine designed to detect structural melodic similarity between MIDI sequences.

---

# 🚀 Core Value

Traditional tools compare audio waveforms.  
Mantra compares melodic structure.

✔ Key-invariant  
✔ Tempo-independent  
✔ Structure-aware  
✔ Regression-tested  

---

# 🧠 Technical Architecture

Pipeline:

MIDI → Pitch Extraction → Interval Normalization → Fingerprint → Similarity Score

---

# 📦 Installation

Create environment:

Windows:

python -m venv venv  
venv\Scripts\activate  

Install dependencies:

pip install -r requirements.txt

---

# 🧪 Run Tests

pytest

Expected result:

14 passed

---

# 🌐 Run API

uvicorn app.main:app --reload

Open browser:

http://127.0.0.1:8000/docs

---

# 📊 Current Status

✔ Stable similarity engine  
✔ Regression validation  
✔ Modular architecture  
✔ SQLite persistence  
✔ FastAPI integration  

---

# 💼 Commercial Direction

• Plagiarism detection SaaS  
• Composer similarity analytics  
• Copyright verification API  
• DAW plugin integration  
• Label & publisher tools  

---

# 📜 License

MIT License