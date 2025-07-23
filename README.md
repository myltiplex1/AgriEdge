# AgriEdge: Smart Farm Assistant

A command-line AI assistant that uses real-time sensor data and textbook-based agricultural knowledge to provide insights, analysis, and actionable suggestions for small to medium-scale farms.

---

## 🚀 Features

- Analyzes **real-time farm sensor data** (soil, water, environment)
- Retrieves context from **agricultural PDF documents**
- Uses **retrieval-augmented generation (RAG)** for knowledge grounding
- Supports **Ollama + LLaMA 3** as the backend LLM
- Produces **human-readable summaries and advice**
- Fully local, privacy-respecting

---

## 📁 Project Structure

```bash
smartfarm/
├── main.py                       # Entry point: command-line interface
├── llm/
│   ├── ollama_llm.py            # LLM query handler using prompt + sensor + RAG
│   └── rag_pipeline.py          # Vector store build & retrieval pipeline
├── logger.py                    # Custom logger setup
├── prompt.txt                   # LLM prompt template
├── data/
│   ├── farm_data_log.json       # JSON log of sensor data
│   ├── docs/                    # Folder for PDF documents (RAG source)
│   └── faiss_index/             # FAISS vector index (auto-generated)
