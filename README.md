# Salary_Pred_Chatbot

## 🚀 Project Overview

This project is designed to help both **job candidates** and **hiring managers**:

* **Candidates**: Estimate a realistic salary range based on their background, role, and location.
* **Hiring Managers**: Determine fair base salaries and overall compensation packages.

---

## 📌 Development Roadmap

### **Phase 1: Salary Prediction Model (MVP)** ✅

* Build a machine learning model for salary range prediction.
* Dataset(From Kaggle): https://www.kaggle.com/datasets/iamsouravbanerjee/software-professional-salaries-2022/data
* Input features: candidate background, role, location, job description, etc.
* Output: expected salary range.

### **Phase 2: LLM-Powered Agent** ✨

* Integrate an LLM agent to provide richer, contextual insights.
* Consider additional inputs such as:
  * Time of application (timestamp data).
  * Relocation requirements.
  * Role-specific requirements.
* Goal: make predictions more **context-aware**.

### **Phase 3: Community Insights (Future Work)** 🔍

* Expand the LLM agent to incorporate **community-driven data** (e.g., Reddit, Glassdoor).
* Example:
  > “Based on similar profiles and postings, applicants with comparable experience tend to receive offers in the range of \$X–\$Y.”

---

## ⚙️ Installation & Running the App

### 1️⃣ Create Environment (via Conda)

```bash
# Clone this repository
git clone https://github.com/niconicoo0w0/Salary_Pred_Chatbot.git
cd Salary_Pred_Chatbot

# Create and activate Conda environment
conda create -n salary_chatbot python=3.10 -y
conda activate salary_chatbot
```
---

### 2️⃣ Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```
---

### 3️⃣ Configure Settings (via `utils/config.py`)

All runtime and model settings are centralized in **`utils/config.py`**
You can edit parameters directly in the `Config` dataclass or override them using environment variables if desired.

Example excerpt:

```python
# utils/config.py
from dataclasses import dataclass
import os

@dataclass
class Config:
    PIPELINE_PATH: str = os.getenv("PIPELINE_PATH", "models/pipeline_new.pkl")
    SCHEMA_PATH: str = os.getenv("SCHEMA_PATH", "models/schema.json")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # OpenAI configuration (optional for LLM explanation)
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

    # Request and cache
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "12"))
    ENABLE_CACHE: bool = os.getenv("ENABLE_CACHE", "true").lower() == "true"
```

You can modify these values directly in the file or through environment overrides, for example:

```bash
export PIPELINE_PATH=models/pipeline_new.pkl
export LOG_LEVEL=DEBUG
```

---

### 4️⃣ Run the App (Gradio Web Interface)

```bash
python app/app.py
```

Then open the local link shown in your terminal, usually:

> 🌐 http://127.0.0.1:7860

You’ll see a web UI for salary prediction.

- You can paste a **Job Description (JD)** and leave job title/location blank — the model will try to parse them.  
- Optionally enable “Use Web Agent” to auto-fill company information (Wikipedia / website / DDG).  
  > Disable this when running offline or during testing.
---

### 5️⃣ Run Tests (Optional)

```bash
pytest -q
```

> To check coverage:
> ```bash
> ./tests/check_coverage.sh
> ```

---

### 6️⃣ (Optional) Retrain the Model

If you want to retrain the model pipeline:

```bash
python models/training_script/train_pipeline.py \
  data/salary_data_cleaned.csv \
  --out models/pipeline_new.pkl 

# Then relaunch
python app/app.py
```

The app automatically validates schema consistency with training outputs.

---

### ✅ Summary

| Step | Command | Description |
|------|----------|--------------|
| 1 | `conda create -n salary_chatbot python=3.10` | Create environment |
| 2 | `pip install -r requirements.txt` | Install dependencies |
| 3 | `python app/app.py` | Run Gradio UI |
| 4 | `pytest` | Run tests |
| 5 | *(Optional)* Train new model | Rebuild salary predictor |
