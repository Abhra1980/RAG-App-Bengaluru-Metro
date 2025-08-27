import os

def read_text_file(dir_path: str, filename: str, encoding: str = "utf-8"):
    file_path = os.path.join(dir_path, filename)
    
    with open(file_path, "r", encoding=encoding) as f:
        content = f.read()
    
    return content