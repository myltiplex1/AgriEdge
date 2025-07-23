import json
from llm.ollama_llm import query_ollama
from llm.rag_pipeline import retrieve_context
from logger import get_logger

logger = get_logger(__name__)

def get_latest_sensor_data(path="data/farm_data_log.json", num_entries=3):
    try:
        with open(path, "r") as f:
            data = json.load(f)
            return data[-num_entries:] if data else []
    except FileNotFoundError:
        logger.error(f"Sensor data file {path} not found.")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {path}: {e}")
        return []

def main():
    logger.info("Smart Farm Assistant started.")
    while True:
        user_query = input("> ")
        if user_query.lower() == 'exit':
            logger.info("Exiting Smart Farm Assistant.")
            break
        logger.info("User query: %s", user_query)
        logger.info("Retrieving latest sensor data...")
        sensor_data_entries = get_latest_sensor_data()
        combined_sensor_data = {
            entry["timestamp"]: {
                "soil": entry["soil"],
                "water": entry["water"],
                "environment": entry["environment"]
            }
            for entry in sensor_data_entries
        }
        try:
            rag_context = retrieve_context(user_query)
            response = query_ollama(user_query, combined_sensor_data, rag_context)
            logger.info("\n--- FARM ASSISTANT RESPONSE ---\n")
            print(response)
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            print("Error: Could not process query. Please try again.")

if __name__ == "__main__":
    main()