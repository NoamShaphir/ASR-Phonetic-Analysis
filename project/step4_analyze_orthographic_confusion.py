import os
import csv
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 4: Analyze Orthographic Confusions (A-Z) ---
# Extracts character-level orthographic errors using Levenshtein distance
# to align standard letters A-Z of baseline words with predicted spellings.
# Computes 26x26 confusion matrices and generates normalized heatmaps
# (both full and error-only variants) stored in project/plots/.

script_dir = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(script_dir, "outputs", "aligned_transcription_report.csv")
PLOTS_DIR = os.path.join(script_dir, "plots")
ARTIFACTS_DIR = r"C:\Users\Noam\.gemini\antigravity\brain\aa679829-b636-404e-b022-7bdd75cf73ad"

os.makedirs(PLOTS_DIR, exist_ok=True)

# English Alphabet character list
ALPHABET = [chr(i) for i in range(65, 91)]
char_to_idx = {char: idx for idx, char in enumerate(ALPHABET)}

# Character Levenshtein alignment
def align_characters(ref_str, hyp_str):
    ref_clean = "".join([c for c in ref_str.upper() if c.isalpha() or c.isspace()])
    hyp_clean = "".join([c for c in hyp_str.upper() if c.isalpha() or c.isspace()])
    
    n, m = len(ref_clean), len(hyp_clean)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_clean[i-1] == hyp_clean[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,
                           dp[i][j-1] + 1,
                           dp[i-1][j-1] + cost)
                           
    i, j = n, m
    pairs = []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref_clean[i-1] == hyp_clean[j-1] else 1
            if dp[i][j] == dp[i-1][j-1] + cost:
                pairs.append((ref_clean[i-1], hyp_clean[j-1]))
                i -= 1; j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i-1][j] + 1:
            pairs.append((ref_clean[i-1], "***"))
            i -= 1
        else:
            pairs.append(("***", hyp_clean[j-1]))
            j -= 1
            
    pairs.reverse()
    return pairs

print(f"Reading CSV aligned report from: {CSV_PATH}")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Missing input aligned report! Please run Step 1 first. Expected path: {CSV_PATH}")

matrix_ru = np.zeros((26, 26))
matrix_en = np.zeros((26, 26))

with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    
    ref_list = []
    pred_list = []
    
    for row in reader:
        if not row:
            continue
        speaker = row[0]
        row_type = row[1]
        
        lang = "Russian" if "russian" in speaker else "English"
        
        if row_type == "Truth":
            ref_list = [w for w in row[2:] if w != "***"]
        elif row_type == "Pred_Opt1":
            # Extract Option 1 predictions (discarding confidence scores)
            pred_list = []
            for item in row[2:]:
                if item != "***" and item.strip():
                    w = item.split()[0]
                    pred_list.append(w)
            
            # Combine into full sentences for character analysis
            truth_text = " ".join(ref_list)
            pred_text = " ".join(pred_list)
            
            pairs = align_characters(truth_text, pred_text)
            
            active_matrix = matrix_ru if lang == "Russian" else matrix_en
            for c_ref, c_hyp in pairs:
                if c_ref in char_to_idx and c_hyp in char_to_idx:
                    idx_ref = char_to_idx[c_ref]
                    idx_hyp = char_to_idx[c_hyp]
                    active_matrix[idx_ref, idx_hyp] += 1

print("Character alignment complete. Generating A-Z heatmaps...")

# Plotting function for 26x26 confusion heatmaps
def plot_heatmap(matrix, title, filename, exclude_diagonal=False):
    norm_matrix = np.zeros_like(matrix)
    
    if exclude_diagonal:
        temp_matrix = matrix.copy()
        np.fill_diagonal(temp_matrix, 0)
        for i in range(26):
            row_sum = temp_matrix[i].sum()
            if row_sum > 0:
                norm_matrix[i] = (temp_matrix[i] / row_sum) * 100
        cmap = "YlOrRd"
        cbar_label = "Error Substitution Rate (%)"
    else:
        for i in range(26):
            row_sum = matrix[i].sum()
            if row_sum > 0:
                norm_matrix[i] = (matrix[i] / row_sum) * 100
        cmap = "Blues"
        cbar_label = "Prediction Rate (%)"
        
    df_plot = pd.DataFrame(norm_matrix, index=ALPHABET, columns=ALPHABET)
    
    plt.figure(figsize=(14, 11))
    ax = sns.heatmap(df_plot, annot=True, fmt=".1f", cmap=cmap, cbar_kws={'label': cbar_label},
                     annot_kws={"size": 8}, linewidths=0.5, square=True)
    
    plt.title(title, pad=20, weight='bold', fontsize=16)
    plt.xlabel("Wav2Vec2 Predicted Character", labelpad=12, weight='bold', fontsize=12)
    plt.ylabel("Ground Truth Character", labelpad=12, weight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300)
    if os.path.exists(ARTIFACTS_DIR):
        plt.savefig(os.path.join(ARTIFACTS_DIR, filename), dpi=300)
    plt.close()

# Save A-Z heatmaps
plot_heatmap(matrix_ru, "Character Confusion Matrix (Normalized %)\nRussian Speakers Accent Group", "confusion_heatmap_ru.png")
plot_heatmap(matrix_en, "Character Confusion Matrix (Normalized %)\nNative English Speakers Group", "confusion_heatmap_en.png")
plot_heatmap(matrix_ru, "ASR Error Substitution Patterns (Excluding Matches %)\nRussian Speakers Accent Group", "error_heatmap_ru.png", exclude_diagonal=True)
plot_heatmap(matrix_en, "ASR Error Substitution Patterns (Excluding Matches %)\nNative English Speakers Group", "error_heatmap_en.png", exclude_diagonal=True)

print(f"\n[+] Orthographic A-Z confusion heatmaps successfully saved!")
print(f"  -> Heatmaps saved inside: {PLOTS_DIR}/")
