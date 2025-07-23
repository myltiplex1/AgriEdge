import streamlit as st
import json
import unicodedata
from llm.ollama_llm import query_ollama
from llm.rag_pipeline import retrieve_context
from logger import get_logger

logger = get_logger(__name__)

def clean_text(text):
    """Normalize Unicode and replace 'None' with cleaner alternative."""
    if isinstance(text, str):
        if text.strip().lower() == "none":
            return "No rainfall"
        return unicodedata.normalize("NFKC", text)
    return text

def get_latest_sensor_data(path="data/farm_data_log.json", num_entries=3):
    try:
        with open(path, "r", encoding="utf-8") as f:  # ✅ UTF-8 decoding
            data = json.load(f)
            return data[-num_entries:] if data else []
    except FileNotFoundError:
        logger.error(f"Sensor data file {path} not found.")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {path}: {e}")
        return []

def run_app():
    st.set_page_config(page_title="Smart Farm Assistant", layout="wide")
    st.title("🌾 Smart Farm Assistant")
    st.markdown("Get AI-powered insights from your farm's sensor data.")

    # Session state setup
    if "history" not in st.session_state:
        st.session_state.history = []

    # Load sensor data
    sensor_data_entries = get_latest_sensor_data()
    if not sensor_data_entries:
        st.error("No sensor data found. Please ensure 'data/farm_data_log.json' exists and contains valid data.")
        return

    # --- 🔁 Conversation History (Top)
    if st.session_state.history:
        st.markdown("## 🗂️ Previous Conversations")
        for idx, (q, r) in enumerate(reversed(st.session_state.history), 1):
            with st.expander(f"🧠 Conversation {idx}: {q}"):
                st.markdown(f"**You asked:** {q}")
                st.markdown(f"**Assistant:** {r}")

    # --- 🧹 Clear history button
    if st.session_state.history:
        if st.button("🧹 Clear Conversation History"):
            st.session_state.history.clear()
            st.success("Conversation history cleared.")
            st.stop()

    # --- 🔍 Sensor data (collapsed)
    with st.expander("📊 Latest Sensor Data Summary", expanded=False):
        for entry in sensor_data_entries:
            st.markdown(f"**🕒 Timestamp:** `{entry['timestamp']}`")
            st.json({
                "soil": {k: clean_text(v) for k, v in entry["soil"].items()},
                "water": {k: clean_text(v) for k, v in entry["water"].items()},
                "environment": {k: clean_text(v) for k, v in entry["environment"].items()}
            })

    # --- 💬 Question input box
    st.markdown("## 💬 Enter your farm-related question:")
    user_query = st.text_input("", placeholder="e.g., What is the current soil condition?")

    if st.button("Analyze") and user_query:
        st.info("Processing your query...")

        combined_sensor_data = {
            entry["timestamp"]: {
                "soil": entry["soil"],
                "water": entry["water"],
                "environment": entry["environment"]
            } for entry in sensor_data_entries
        }

        try:
            rag_context = retrieve_context(user_query)
            response = query_ollama(user_query, combined_sensor_data, rag_context)

            # Display result immediately
            st.success("✅ Analysis Complete")
            st.markdown("### ✅ Latest Response")
            st.markdown(f"**You asked:** {user_query}")
            st.markdown("### 🧑‍🌾 AgriEdge:")
            st.markdown(response)

            # Save to history (limit to 5)
            st.session_state.history.append((user_query, response))
            st.session_state.history = st.session_state.history[-5:]

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            st.error("An error occurred while processing your query. Please try again.")


if __name__ == "__main__":
    run_app()
