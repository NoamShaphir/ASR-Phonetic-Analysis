import os
import csv
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import eng_to_ipa as ipa

# --- Step 3: Analyze Phonetic Confusions (IPA-to-IPA) ---
# Compares character-level spoken broad IPA with G2P-decoded predicted IPA.
# Aligns individual phonemes character-by-character using Levenshtein distance.
# Computes 23x23 IPA-to-IPA confusion matrices for Russian and English groups.
# Generates normalized heatmap plots (both full and error-only) saved in project/plots/.
# Merges get_errors.py logic to output a detailed phonetic error list to outputs/phonetic_errors.txt.

script_dir = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(script_dir, "outputs", "aligned_transcription_report.csv")
PLOTS_DIR = os.path.join(script_dir, "plots")
OUTPUT_ERRORS_PATH = os.path.join(script_dir, "outputs", "phonetic_errors.txt")
ARTIFACTS_DIR = r"C:\Users\Noam\.gemini\antigravity\brain\aa679829-b636-404e-b022-7bdd75cf73ad"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_ERRORS_PATH), exist_ok=True)

# 23 Core Phonemes appearing in the elicitation baseline text
IPA_PHONEMES = [
    'p', 'b', 't', 'd', 'k', 'g', 'f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ɹ', 'l', 'w', 'j', 'i', 'ɪ', 'ɛ', 'æ', 'ə', 'ʌ'
]
phoneme_to_idx = {ph: idx for idx, ph in enumerate(IPA_PHONEMES)}

# Dictionary cache to accelerate G2P API calls up to 100x
ipa_cache = {}

def word_to_ipa(word):
    word = word.lower().strip()
    if not word or word == "***":
        return "***"
    
    if word in ipa_cache:
        return ipa_cache[word]
    
    # dictionary-based G2P conversion
    ipa_word = ipa.convert(word)
    
    # If the word is out of dictionary vocabulary (OOV, eng_to_ipa returns with asterisk)
    if '*' in ipa_word:
        clean_word = ipa_word.replace('*', '')
        # Direct rule-based orthographic-to-phonetic conversions for spelling slips
        g2p_rules = [
            ('sh', 'ʃ'), ('ch', 'tʃ'), ('th', 'θ'), ('ph', 'f'),
            ('ee', 'i'), ('oo', 'u'), ('ea', 'i'), ('oy', 'ɔɪ'), ('oi', 'ɔɪ'),
            ('ay', 'eɪ'), ('ai', 'eɪ'), ('ow', 'oʊ'), ('ou', 'aʊ'),
            ('ck', 'k'), ('qu', 'kw'),
            ('c', 'k'), ('x', 'ks'), ('y', 'j'),
            ('a', 'æ'), ('e', 'ɛ'), ('i', 'ɪ'), ('o', 'ɑ'), ('u', 'ʌ'),
            ('r', 'ɹ'), ('j', 'dʒ'), ('ɡ', 'g')
        ]
        temp = clean_word
        for pattern, replacement in g2p_rules:
            temp = temp.replace(pattern, replacement)
        ipa_word = temp
    else:
        # Strip extraneous stress, elongation, and unicode combining marks
        ipa_word = re.sub(r'[ˈˌʲʷ̚ʰː:]', '', ipa_word)
        ipa_word = ipa_word.replace('r', 'ɹ')
        ipa_word = ipa_word.replace('ɡ', 'g')
        
    ipa_cache[word] = ipa_word
    return ipa_word

# Custom phonetic substitution distance cost matrix
def ipa_to_ipa_similarity(char1, char2):
    c1, c2 = char1.lower(), char2.lower()
    if c1 == c2:
        return 0.0
    
    # Highly similar phonetic pairs (lower substitution cost of 0.2-0.3)
    similar_pairs = [
        ({'i', 'ɪ'}, 0.2), # tense/lax close vowels
        ({'ɛ', 'æ'}, 0.2), # front open-mid/open vowels
        ({'ə', 'ʌ'}, 0.2), # central schwa/lax open-mid vowels
        ({'s', 'z'}, 0.2), # alveolar fricatives voicing contrast
        ({'t', 'd'}, 0.2), # alveolar stops voicing contrast
        ({'p', 'b'}, 0.2), # bilabial stops voicing contrast
        ({'k', 'g'}, 0.2), # velar stops voicing contrast
        ({'f', 'v'}, 0.2), # labiodental fricatives voicing contrast
        ({'θ', 'ð'}, 0.2), # dental fricatives voicing contrast
        ({'ʃ', 's'}, 0.3), # palato-alveolar / alveolar fricatives
        ({'w', 'v'}, 0.3), # labiodental/bilabial fricative/approximant confusion
        ({'ɹ', 'l'}, 0.3)  # liquid consonants contrast
    ]
    for pair, cost in similar_pairs:
        if c1 in pair and c2 in pair:
            return cost
            
    return 1.0

# Levenshtein character-by-character IPA alignment
def align_ipa_to_predicted(spoken_ipa, pred_ipa):
    spoken_clean = "".join([c for c in spoken_ipa.lower() if c in IPA_PHONEMES or c.isspace()])
    pred_clean = "".join([c for c in pred_ipa.lower() if c in IPA_PHONEMES or c.isspace()])
    
    n, m = len(spoken_clean), len(pred_clean)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i * 1.0
    for j in range(m + 1): dp[0][j] = j * 1.0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = ipa_to_ipa_similarity(spoken_clean[i-1], pred_clean[j-1])
            dp[i][j] = min(dp[i-1][j] + 1.0,
                           dp[i][j-1] + 1.0,
                           dp[i-1][j-1] + cost)
                           
    i, j = n, m
    pairs = []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = ipa_to_ipa_similarity(spoken_clean[i-1], pred_clean[j-1])
            if abs(dp[i][j] - (dp[i-1][j-1] + cost)) < 1e-4:
                pairs.append((spoken_clean[i-1], pred_clean[j-1]))
                i -= 1; j -= 1
                continue
        if i > 0 and abs(dp[i][j] - (dp[i-1][j] + 1.0)) < 1e-4:
            pairs.append((spoken_clean[i-1], "***"))
            i -= 1
        else:
            pairs.append(("***", pred_clean[j-1]))
            j -= 1
            
    pairs.reverse()
    return pairs

print(f"Loading report from: {CSV_PATH}")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Missing input aligned report! Please run Step 1 first. Expected path: {CSV_PATH}")

matrix_ru = np.zeros((len(IPA_PHONEMES), len(IPA_PHONEMES)))
matrix_en = np.zeros((len(IPA_PHONEMES), len(IPA_PHONEMES)))

with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    ipa_row = []
    
    for row in reader:
        if not row:
            continue
        speaker = row[0]
        row_type = row[1]
        
        lang = "Russian" if "russian" in speaker else "English"
        
        if row_type == "IPA":
            ipa_row = [w for w in row[2:] if w != "***" and w != "[No IPA]"]
        elif row_type == "Pred_Opt1":
            pred_row = []
            for item in row[2:]:
                if item != "***" and item.strip():
                    w = item.split()[0]
                    pred_row.append(w)
            
            # IPA character-level Levenshtein alignment
            if ipa_row and pred_row:
                pred_ipa_row = [word_to_ipa(w) for w in pred_row]
                
                spoken_ipa_text = " ".join(ipa_row)
                pred_ipa_text = " ".join(pred_ipa_row)
                
                pairs = align_ipa_to_predicted(spoken_ipa_text, pred_ipa_text)
                
                active_matrix = matrix_ru if lang == "Russian" else matrix_en
                for spoken_phoneme, pred_phoneme in pairs:
                    if spoken_phoneme == 'ɡ': spoken_phoneme = 'g'
                    if pred_phoneme == 'ɡ': pred_phoneme = 'g'
                    
                    if spoken_phoneme in phoneme_to_idx and pred_phoneme in phoneme_to_idx:
                        idx_spoken = phoneme_to_idx[spoken_phoneme]
                        idx_pred = phoneme_to_idx[pred_phoneme]
                        active_matrix[idx_spoken, idx_pred] += 1

print("Phoneme alignment complete. Creating heatmaps...")

# Plotting function for phonetic matrices
def plot_phonetic_heatmap(matrix, title, filename, exclude_correct=False):
    norm_matrix = np.zeros_like(matrix)
    temp_matrix = matrix.copy()
    
    if exclude_correct:
        # Clear diagonal elements to show substitutions only
        np.fill_diagonal(temp_matrix, 0)
        for i in range(len(IPA_PHONEMES)):
            row_sum = temp_matrix[i].sum()
            if row_sum > 0:
                norm_matrix[i] = (temp_matrix[i] / row_sum) * 100
        cmap = "Reds"
        cbar_label = "Phonetic Substitution Rate (excluding exact matches %)"
    else:
        for i in range(len(IPA_PHONEMES)):
            row_sum = matrix[i].sum()
            if row_sum > 0:
                norm_matrix[i] = (matrix[i] / row_sum) * 100
        cmap = "Oranges"
        cbar_label = "Prediction Rate (Normalized %)"
        
    df_plot = pd.DataFrame(norm_matrix, index=IPA_PHONEMES, columns=IPA_PHONEMES)
    
    plt.figure(figsize=(15, 11))
    ax = sns.heatmap(df_plot, annot=True, fmt=".1f", cmap=cmap, cbar_kws={'label': cbar_label},
                     annot_kws={"size": 8}, linewidths=0.5, square=True)
    
    plt.title(title, pad=20, weight='bold', fontsize=16)
    plt.xlabel("Wav2Vec2 Predicted Phoneme (IPA Sound)", labelpad=12, weight='bold', fontsize=12)
    plt.ylabel("Spoken IPA Phoneme (Sound)", labelpad=12, weight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300)
    if os.path.exists(ARTIFACTS_DIR):
        plt.savefig(os.path.join(ARTIFACTS_DIR, filename), dpi=300)
    plt.close()

# Generate Heatmaps
plot_phonetic_heatmap(matrix_ru, "Phonetic Sound Confusion Matrix (Normalized %)\nRussian Speakers Accent Group (IPA-to-IPA)", "phonetic_heatmap_ru.png")
plot_phonetic_heatmap(matrix_en, "Phonetic Sound Confusion Matrix (Normalized %)\nNative English Speakers Group (IPA-to-IPA)", "phonetic_heatmap_en.png")
plot_phonetic_heatmap(matrix_ru, "Phonetic Error Substitutions (Excluding Diagonal Matches %)\nRussian Speakers Accent Group (IPA-to-IPA)", "phonetic_error_heatmap_ru.png", exclude_correct=True)
plot_phonetic_heatmap(matrix_en, "Phonetic Error Substitutions (Excluding Diagonal Matches %)\nNative English Speakers Group (IPA-to-IPA)", "phonetic_error_heatmap_en.png", exclude_correct=True)

# Generate detailed phonetic errors text report
print(f"Generating phonetic errors stats report at: {OUTPUT_ERRORS_PATH}")

with open(OUTPUT_ERRORS_PATH, "w", encoding="utf-8") as out:
    out.write("========================================================================\n")
    out.write("           DETAILED PHONETIC SUBSTITUTIONS STATS (IPA-TO-IPA)           \n")
    out.write("========================================================================\n\n")
    
    out.write("--- Top Phonetic substitutions (Voicing/Acoustic transfers) for Russian Speakers ---\n")
    for i, ipa_ph in enumerate(IPA_PHONEMES):
        row = matrix_ru[i].copy()
        row[i] = 0  # clear exact diagonal matches
        row_sum = row.sum()
        if row_sum > 0:
            top_err_indices = np.argsort(row)[::-1][:3]
            err_str = ", ".join([f"/{IPA_PHONEMES[idx]}/ ({(row[idx]/row_sum*100):.1f}%)" for idx in top_err_indices if row[idx] > 0])
            out.write(f"Spoken /{ipa_ph}/ -> Predicted: {err_str}\n")
            
    out.write("\n--- Top Phonetic substitutions for Native English Speakers ---\n")
    for i, ipa_ph in enumerate(IPA_PHONEMES):
        row = matrix_en[i].copy()
        row[i] = 0  # clear exact diagonal matches
        row_sum = row.sum()
        if row_sum > 0:
            top_err_indices = np.argsort(row)[::-1][:3]
            err_str = ", ".join([f"/{IPA_PHONEMES[idx]}/ ({(row[idx]/row_sum*100):.1f}%)" for idx in top_err_indices if row[idx] > 0])
            out.write(f"Spoken /{ipa_ph}/ -> Predicted: {err_str}\n")

print(f"\n[+] Phonetic confusion heatmaps and error list successfully saved!")
print(f"  -> Heatmaps saved inside: {PLOTS_DIR}/")
print(f"  -> Error Stats saved at: {OUTPUT_ERRORS_PATH}")
