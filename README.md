# 🌾 AgriEdge: Smart Farm Assistant

An AI-powered assistant that uses real-time sensor data and textbook-based agricultural knowledge to provide insights, analysis, and actionable suggestions for small to medium-scale farms. Comes with both command-line and web interfaces.

---

## 🚀 Features

- 📡 Analyzes **real-time farm sensor data** (soil, water, environment)
- 📚 Retrieves context from **agricultural PDF documents**
- 🤖 Uses **retrieval-augmented generation (RAG)** for grounded reasoning
- 🧠 Powered by **Ollama + LLaMA 3**
- 📝 Generates **natural language summaries and actionable insights**
- 🔒 Runs **fully local** — no cloud, no data sharing
- 🌐 Supports **Streamlit-based dashboard** for non-technical users

---

## 📁 Project Structure

```bash
smartfarm/
├── main.py                       # Command-line interface
├── app.py                        # Streamlit web app
├── llm/
│   ├── ollama_llm.py             # Query handler using LLM + sensor data + RAG
│   └── rag_pipeline.py           # PDF retrieval pipeline using FAISS
├── logger.py                     # Logging setup
├── prompt.txt                    # Prompt template for LLM
├── data/
│   ├── farm_data_log.json        # JSON file logging sensor readings
│   ├── docs/                     # Agricultural PDFs for knowledge retrieval
│   └── faiss_index/              # Auto-generated FAISS vector index
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/smartfarm.git
cd smartfarm
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Ollama

Make sure you have Ollama installed and running.

Download the LLaMA 3 model:

```bash
ollama run llama3
```

Make sure Ollama is running in the background before using the assistant.

### 4. Add Sensor Data

Append new entries to `data/farm_data_log.json`. Example format:

```json
{
  "timestamp": "2025-07-22T21:00:00+01:00",
  "soil": {"moisture": "High", "pH": 6.8, "temperature": 24.9},
  "water": {"pH": 7.2, "turbidity": "8 NTU", "temperature": 23.3},
  "environment": {"humidity": "85%", "temperature": 26.0, "rainfall": "Moderate"}
}
```

### 5. Add Agricultural Documents (Optional)

Place your farming-related PDFs inside:

```bash
data/docs/
```

The system will automatically build a searchable vector index.

---

## ▶️ Usage

### 📟 Command-Line Mode

```bash
python main.py
```

You’ll be prompted to enter queries like:

```markdown
> Is the soil suitable for planting now?
> Has the turbidity improved compared to earlier?
```

Type `exit` to quit.

### 🌐 Streamlit Web Interface

Launch the UI with:

```bash
streamlit run app.py
```

What you can do:

- View the most recent sensor snapshot
- Ask farm-related questions like:
  - "What is the current soil condition?"
  - "Is it safe to irrigate now?"
  - "Has rainfall increased compared to earlier?"

---

## 💡 Notes

- The system analyzes only the most recent sensor reading but uses the previous 2 for historical comparison (internally).
- No internet connection is required once the vector store and model are set up.
- Logs are written automatically to `logs/`.

---

## 🧪 Example Questions

- "Is the soil moisture improving?"
- "What is the overall environmental condition right now?"
- "Is the water quality good for irrigation?"

---

## 📄 License

MIT License
