import os 
import scanpy as sc
import re
import json

def get_data_path():
    
    while True:
        data_file_path = input("Please enter the path of your scRNA seq data file (enter 'q' to exit):\n").strip()

        if data_file_path.lower() == 'q':  
            print("The user has cancelled the input.")
            return None

        if os.path.exists(data_file_path) and data_file_path.endswith(".h5ad"):
            return data_file_path
        else:
            print(f"The data file `{data_file_path}` does not exist or has an incorrect format. Please re-enter!")

def read_scrna_data(file_path):

    try:
        adata = sc.read_h5ad(file_path)
        print("read data success!")
        return adata
    except Exception as e:
        print(f"read data failed: {e}")
        return None
    
def extract_json_from_response(response):
    
    try:
        code_blocks = re.findall(r"```json(.*?)```", response, re.DOTALL)
        if code_blocks:
            return json.loads(code_blocks[0].strip())
        else:
            json_like = re.search(r"\{.*\}", response, re.DOTALL)
            return json.loads(json_like.group()) if json_like else None
    except Exception as e:
        print(f"JSON parsing failed: {e}")
        return None