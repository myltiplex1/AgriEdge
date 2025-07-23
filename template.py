import os

# Define directory structure
project_structure = {
    "sensors": [
        "sensor_collector.py"
    ],
    "llm": [
        "ollama_llm.py",
        "rag_pipeline.py"
    ],
    "data": [
        "farm_data_log.json",
        "docs/",           # Folder
        "faiss_index/"     # Folder
    ],
    ".": [                # Root directory
        "main.py",
        "requirements.txt"
    ]
}

def create_structure(base_path="."):
    for folder, items in project_structure.items():
        folder_path = os.path.join(base_path, folder) if folder != "." else base_path
        os.makedirs(folder_path, exist_ok=True)

        for item in items:
            item_path = os.path.join(folder_path, item)
            if item.endswith("/"):
                os.makedirs(item_path, exist_ok=True)
                print(f"Created folder: {item_path}")
            else:
                if not os.path.exists(item_path):
                    with open(item_path, "w") as f:
                        pass
                print(f"Created file: {item_path}")

if __name__ == "__main__":
    create_structure()