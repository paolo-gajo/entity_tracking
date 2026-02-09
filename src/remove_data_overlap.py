import pandas as pd
import json
import os
from rapidfuzz import process, fuzz, utils
from tqdm.auto import tqdm

def load_yamakata_data(file_paths):
    """
    Loads and combines Train/Dev/Test splits from Yamakata (ERFGC) JSON files.
    Joins the 'words' list into a single string for comparison.
    """
    dfs = []
    for path in file_paths:
        if not os.path.exists(path):
            print(f"Warning: File not found: {path}")
            continue
            
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        df_chunk = pd.DataFrame(data)
        # Create a 'text' column for matching
        df_chunk['text'] = df_chunk['words'].apply(lambda x: " ".join(x))
        df_chunk['split_source'] = os.path.basename(path)
        dfs.append(df_chunk)
    
    if not dfs:
        raise FileNotFoundError("No Yamakata files could be loaded.")
        
    return pd.concat(dfs, ignore_index=True)

def load_recipenlg_data(file_path):
    """
    Loads RecipeNLG. Joins 'directions' list into a single string.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"RecipeNLG file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    # RecipeNLG directions are lists of strings. Join them for matching.
    df['text'] = df['directions'].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
    
    return df

def find_matches(source_df, target_df, threshold=90.0):
    print(f"Starting matching process...")
    print(f"Source (Yamakata): {len(source_df)} rows")
    print(f"Target (RecipeNLG): {len(target_df)} rows")

    matches = []
    
    # 1. Pre-calculate lengths
    source_df['len'] = source_df['text'].str.len()
    target_df['len'] = target_df['text'].str.len()
    
    for idx, row in tqdm(source_df.iterrows(), total=len(source_df), desc="finding matches..."):
        src_text = row['text']
        src_len = row['len']
        
        # 2. Length Filter (+/- 20%)
        len_margin = max(10, int(src_len * 0.2)) 
        
        candidates = target_df[
            (target_df['len'] >= src_len - len_margin) & 
            (target_df['len'] <= src_len + len_margin)
        ]
        
        if candidates.empty:
            continue
            
        # 3. Create Dictionary for Index Retrieval
        choices_dict = candidates['text'].to_dict()
        
        best_match = process.extractOne(
            src_text, 
            choices_dict, 
            scorer=fuzz.ratio, 
            score_cutoff=threshold
        )
        
        if best_match:
            match_text, score, match_idx = best_match
            target_row = target_df.loc[match_idx]
            
            matches.append({
                'yamakata_idx': idx,
                'recipenlg_idx': match_idx, # We need this to drop later
                'score': score,
                'yamakata_text': src_text,
                'recipenlg_title': target_row.get('title', 'Unknown')
            })

    return pd.DataFrame(matches)

# --- Execution Block ---

yamakata_files = [
    'data/erfgc/bio/test.json',
    'data/erfgc/bio/train.json',
    'data/erfgc/bio/val.json'
]

recipenlg_file = 'data/recipenlg/recipenlg.json'
output_cleaned_file = 'data/recipenlg/recipenlg_clean.json'
output_matches_csv = 'yamakata_recipenlg_matches.csv'

try:
    # 1. Load Data
    df_yamakata = load_yamakata_data(yamakata_files)
    df_recipenlg = load_recipenlg_data(recipenlg_file)

    # 2. Find Matches (Threshold 90.0)
    print("\n--- Identifying Overlaps ---")
    df_matches = find_matches(df_yamakata, df_recipenlg, threshold=90.0)

    # 3. Process Results
    if not df_matches.empty:
        print(f"\nFound {len(df_matches)} matches with score >= 90.0")
        
        # Save matches for inspection
        df_matches.to_csv(output_matches_csv, index=False)
        print(f"Match details saved to: {output_matches_csv}")

        # --- REMOVAL LOGIC ---
        indices_to_drop = df_matches['recipenlg_idx'].unique()
        
        print(f"Dropping {len(indices_to_drop)} unique records from RecipeNLG...")
        original_count = len(df_recipenlg)
        
        # Drop the rows
        df_recipenlg_clean = df_recipenlg.drop(index=indices_to_drop)
        
        # Cleanup: Remove the helper columns we added ('text' and 'len')
        # so the output JSON has the same schema as the input
        df_recipenlg_clean = df_recipenlg_clean.drop(columns=['text', 'len'], errors='ignore')
        
        new_count = len(df_recipenlg_clean)
        print(f"Done. Rows reduced from {original_count} -> {new_count}")

        # Save Cleaned Data
        print(f"Saving cleaned dataset to {output_cleaned_file}...")
        df_recipenlg_clean.to_json(output_cleaned_file, orient='records', indent=4)
        print("Save complete.")
        
    else:
        print("\nNo matches found above threshold. No files created/modified.")

except Exception as e:
    print(f"An error occurred: {e}")