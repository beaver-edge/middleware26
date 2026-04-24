import os
import glob
import pandas as pd
import code_bert_score
from datetime import datetime
import torch

# Configuration
REFERENCE_FILE = 'experimental_data/processed/similarity/references/object_color_classify.ino'
CANDIDATE_DIR = 'experimental_data/raw/arduino'
OUTPUT_CSV = 'experimental_data/processed/similarity/similarity_results_ino.csv'
LANG = 'c_sharp'

def load_file_content(file_path):
    """Reads the content of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def get_processed_candidates(csv_path):
    """Returns a set of processed candidate paths from the CSV."""
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path)
        if 'Candidate_Path' in df.columns:
            return set(df['Candidate_Path'].tolist())
    except Exception as e:
        print(f"Error reading existing CSV: {e}")
    return set()

def main():
    # 1. Load Reference
    print(f"Loading reference file: {REFERENCE_FILE}")
    reference_content = load_file_content(REFERENCE_FILE)
    if not reference_content:
        print("Failed to load reference file. Exiting.")
        return

    # 2. Identify Candidates
    print(f"Scanning for candidates in: {CANDIDATE_DIR}")
    # Look for .ino files in subdirectories of CANDIDATE_DIR
    # Pattern: compiling/*/valid_*.ino based on file exploration
    # But user said "containing many generated sketch folders, each has a .ino file"
    # So we will search recursively or just one level deep.
    # Based on `list_dir` output: compiling/valid_.../valid_...ino
    candidate_files = glob.glob(os.path.join(CANDIDATE_DIR, '**', '*.ino'), recursive=True)
    
    # Filter out the reference file if it happens to be in the list (unlikely given path)
    candidate_files = [f for f in candidate_files if os.path.abspath(f) != os.path.abspath(REFERENCE_FILE)]
    
    print(f"Found {len(candidate_files)} candidate files.")

    # 3. Resumption
    processed_candidates = get_processed_candidates(OUTPUT_CSV)
    print(f"Already processed {len(processed_candidates)} candidates.")
    
    candidates_to_process = [f for f in candidate_files if f not in processed_candidates]
    print(f"Remaining candidates to process: {len(candidates_to_process)}")

    if not candidates_to_process:
        print("No new candidates to process.")
        return

    # 4. Processing
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Prepare CSV header if file doesn't exist
    if not os.path.exists(OUTPUT_CSV):
        df_header = pd.DataFrame(columns=[
            'Candidate_ID', 'Model','Precision', 'Recall', 'F1', 'F3', 
            'Reference_File', 'Timestamp'
        ])
        df_header.to_csv(OUTPUT_CSV, index=False)

    for candidate_path in candidates_to_process:
        print(f"Processing: {candidate_path}")
        candidate_content = load_file_content(candidate_path)
        
        if not candidate_content:
            print(f"Skipping {candidate_path} due to read error.")
            continue
            
        try:
            # Calculate Score
            # score returns P, R, F1, F3
            # We pass lists of strings
            P, R, F1, F3 = code_bert_score.score(
                cands=[candidate_content], 
                refs=[reference_content], 
                lang=LANG,
                device=device,
                verbose=False
            )
            
            # Extract scalar values
            p_val = P[0].item()
            r_val = R[0].item()
            f1_val = F1[0].item()
            f3_val = F3[0].item()
            
            # Create result record
            candidate_id = os.path.basename(os.path.dirname(candidate_path)) # Folder name as ID
            timestamp = datetime.now().isoformat()
            
            result = {
                'Candidate_ID': candidate_id,
                # 'Candidate_Path': candidate_path,
                'Model': "gpt-4o",
                'Precision': p_val,
                'Recall': r_val,
                'F1': f1_val,
                'F3': f3_val,
                'Reference_File': REFERENCE_FILE,
            
                'Timestamp': timestamp
            }
            
            # Append to CSV
            df_result = pd.DataFrame([result])
            df_result.to_csv(OUTPUT_CSV, mode='a', header=False, index=False)
            
        except Exception as e:
            print(f"Error processing {candidate_path}: {e}")

    print("Processing complete.")

if __name__ == "__main__":
    main()
