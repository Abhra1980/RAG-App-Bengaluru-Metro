import os
import json
import pandas as pd

def save_text_to_csv(text: str, dir_path: str, filename: str = "output.csv"):
    # Ensure the directory exists
    os.makedirs(dir_path, exist_ok=True)
    
    # Build the full file path
    file_path = os.path.join(dir_path, filename)
    
    # Save to CSV
    df = pd.DataFrame({"text": [text]})
    df.to_csv(file_path, index=False)
    
    print(f"✅ File saved at: {file_path}")
    return file_path

def save_text_to_json(text, dir_path: str, filename: str = "output.json"):
    # Ensure the directory exists
    os.makedirs(dir_path, exist_ok=True)

    # Build the full file path
    file_path = os.path.join(dir_path, filename)

    # Wrap string into list if needed
    if isinstance(text, str):
        data = [{"text": text}]
    elif isinstance(text, list):
        data = [{"text": t} for t in text]
    else:
        raise TypeError("text must be a string or a list of strings")

    # Save to JSON
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"✅ JSON file saved at: {file_path}")
    return file_path
