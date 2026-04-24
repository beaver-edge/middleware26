import os
import glob
import pandas as pd
import code_bert_score
from datetime import datetime
import torch

# Configuration
REFERENCE_FILE = 'experimental_data/processed/similarity/references/TFLite_detection_video.py'
CANDIDATE_ROOT = 'experimental_data/raw/raspberrypi'
OUTPUT_CSV = 'experimental_data/processed/similarity/similarity_results_py.csv'
LANG = 'python'

def load_file_content(file_path):
    """Reads the content of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def extract_model_name(directory_name):
    """Extracts model name from directory name (substring after 'psg_')."""
    if 'psg_' in directory_name:
        return directory_name.split('psg_')[1]
    return 'unknown'

def main():
    # 1. Load Reference
    print(f"Loading reference file: {REFERENCE_FILE}")
    reference_content = load_file_content(REFERENCE_FILE)
    if not reference_content:
        print("Failed to load reference file. Exiting.")
        return

    # 2. Identify Candidates
    print(f"Scanning for candidates in: {CANDIDATE_ROOT}")
    # Find all subdirectories in otherSG
    subdirs = [d for d in os.listdir(CANDIDATE_ROOT) if os.path.isdir(os.path.join(CANDIDATE_ROOT, d))]
    
    # Filter for valid candidates
    valid_dirs = [d for d in subdirs if d.startswith('valid') and 'psg' in d]
    print(f"Found {len(valid_dirs)} valid candidate directories.")

    # 3. Processing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Prepare CSV header
    if not os.path.exists(OUTPUT_CSV):
        df_header = pd.DataFrame(columns=[
            'Candidate_ID',   'Model', 'Precision', 'Recall', 'F1', 'F3', 
            'Reference_File',  'Timestamp'
        ])
        df_header.to_csv(OUTPUT_CSV, index=False)
        existing_ids = set()
    else:
        # Load existing IDs to resume
        try:
            df_existing = pd.read_csv(OUTPUT_CSV)
            existing_ids = set(df_existing['Candidate_ID'].astype(str))
            print(f"Resuming... {len(existing_ids)} files already processed.")
        except Exception as e:
            print(f"Error reading existing CSV: {e}. Starting fresh.")
            existing_ids = set()

    for dir_name in valid_dirs:
        if dir_name in existing_ids:
            continue

        dir_path = os.path.join(CANDIDATE_ROOT, dir_name)
        # Find .py file in the directory
        py_files = glob.glob(os.path.join(dir_path, '*.py'))
        
        if not py_files:
            print(f"No .py file found in {dir_name}")
            continue
            
        # Assuming one .py file per valid directory, or take the first one
        candidate_path = py_files[0]
        print(f"Processing: {candidate_path.split('/')[-1]}")
        
        candidate_content = load_file_content(candidate_path)
        if not candidate_content:
            continue
            
        try:
            # Calculate Score
            P, R, F1, F3 = code_bert_score.score(
                cands=[candidate_content], 
                refs=[reference_content], 
                lang=LANG, 
                device=device,
                verbose=False
            )
            
            p_val = P[0].item()
            r_val = R[0].item()
            f1_val = F1[0].item()
            f3_val = F3[0].item()
            
            model_name = extract_model_name(dir_name)
            timestamp = datetime.now().isoformat()
            
            result = {
                'Candidate_ID': dir_name,
                # 'Candidate_Path': candidate_path,
                'Model': model_name,
                'Precision': p_val,
                'Recall': r_val,
                'F1': f1_val,
                'F3': f3_val,
                'Reference_File': REFERENCE_FILE,
                'Timestamp': timestamp
            }
            
            df_result = pd.DataFrame([result])
            df_result.to_csv(OUTPUT_CSV, mode='a', header=False, index=False)
            
        except Exception as e:
            print(f"Error processing {candidate_path}: {e}")

    print("Processing complete.")

if __name__ == "__main__":
    main()
