import pickle
import json
import os
from typing import Any


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def save_model(model,trained_model_path):
    with open(trained_model_path, "wb") as f:
        pickle.dump(model, f)

def write_json_to_file(data: Any, file_path: str) -> bool:
    """
    Writes JSON-serializable data to a file.
    Overwrites file if it exists.
    
    Returns:
        True  -> if write succeeds
        False -> if any error occurs
    """
    try:
        # Ensure directory exists (if directory is provided)
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # Write file in overwrite mode
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        return True

    except TypeError as e:
        print(f"Serialization Error: Data is not JSON serializable -> {e}")
    
    except OSError as e:
        print(f"File System Error: Could not write file -> {e}")
    
    except Exception as e:
        print(f"Unexpected Error: {e}")

    return False
