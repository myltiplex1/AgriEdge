import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from logger import get_logger

logger = get_logger(__name__)

# Load prompt from external file
PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompt.txt")

def load_prompt_template():
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            template = f.read()
        return ChatPromptTemplate.from_template(template)
    except FileNotFoundError:
        logger.error(f"Prompt file {PROMPT_FILE} not found.")
        raise
    except Exception as e:
        logger.error(f"Error loading prompt file: {e}")
        raise

# Initialize model and prompt
llm = ChatOllama(model="llama3.2:3b")
prompt_template = load_prompt_template()

def query_ollama(user_query: str, sensor_data: dict = None, context: str = "") -> str:
    logger.info("Querying LLM with user query and context.")
    if sensor_data:
        # Use only the latest timestamp
        # Sort timestamps in ascending order
        timestamps = sorted(sensor_data.keys())[-3:]  # Last 3 entries

        sensor_lines = ["Sensor Data (latest 3 readings):"]
        for ts in timestamps:
            values = sensor_data[ts]
            sensor_lines.append(f"\nTimestamp: {ts}")
            for category, readings in values.items():
                sensor_lines.append(f"{category.capitalize()}:")
                for metric, val in readings.items():
                    sensor_lines.append(f"  {metric}: {val}")
        
        # Now summarize changes between timestamps
        sensor_lines.append("\nTrends and Changes Summary:")
        try:
            if len(timestamps) == 3:
                t1, t2, t3 = timestamps  # oldest to newest
                for category in ["soil", "water", "environment"]:
                    sensor_lines.append(f"\n{category.capitalize()} Trends:")
                    for metric in sensor_data[t3][category]:
                        try:
                            v1 = sensor_data[t1][category][metric]
                            v2 = sensor_data[t2][category][metric]
                            v3 = sensor_data[t3][category][metric]

                            # Only compare numeric values
                            def to_float(v):
                                return float(str(v).replace("°C", "").replace("NTU", "").replace("%", "").strip())

                            fv1, fv2, fv3 = to_float(v1), to_float(v2), to_float(v3)
                            trend = "increased" if fv3 > fv1 else "decreased" if fv3 < fv1 else "remained stable"
                            sensor_lines.append(f"  {metric}: {trend} (from {v1} → {v2} → {v3})")
                        except:
                            # If conversion fails (non-numeric), just show as categorical
                            trend = "changed" if v1 != v3 else "unchanged"
                            sensor_lines.append(f"  {metric}: {trend} (from '{v1}' → '{v2}' → '{v3}')")
        except Exception as e:
            logger.warning(f"Trend analysis failed: {e}")
        
        sensor_status = "\n".join(sensor_lines)


    # Fill prompt
    try:
        messages = prompt_template.format_messages(
            sensor_status=sensor_status,
            rag_context=context,
            user_query=user_query
        )
        response = llm.invoke(messages)
        logger.info("Received response from LLM.")
        return response.content  # Extract content from response
    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        return f"Error: Could not process query. Please try again."