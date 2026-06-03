import os
import csv
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 2: Analyze ASR Word Error Rates (WER) ---
# Parses the aligned CSV report, calculates overall Standard and Oracle WER metrics,
# extracts failure statistics for each of the 69 words in the baseline text,
# and generates premium comparative visualizations stored in project/plots/.

script_dir = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(script_dir, "outputs", "aligned_transcription_report.csv")
PLOTS_DIR = os.path.join(script_dir, "plots")
ARTIFACTS_DIR = r"C:\Users\Noam\.gemini\antigravity\brain\aa679829-b636-404e-b022-7bdd75cf73ad"

os.makedirs(PLOTS_DIR, exist_ok=True)

# Ground Truth Raw Elicitation Paragraph
GROUND_TRUTH_RAW = (
    "Please call Stella.  Ask her to bring these things with her from the store:  "
    "Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.  "
    "We also need a small plastic snake and a big toy frog for the kids.  "
    "She can scoop these things into three red bags, and we will go meet her Wednesday at the train station."
)
GROUND_TRUTH_CLEAN = re.sub(r"[^A-Z'\s]", " ", GROUND_TRUTH_RAW.upper())
GROUND_TRUTH_WORDS = GROUND_TRUTH_CLEAN.split()

print(f"Reading aligned CSV report from: {CSV_PATH}")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Missing input aligned report! Please run Step 1 first. Expected path: {CSV_PATH}")

speakers_data = []
word_errors_russian = {w: 0 for w in GROUND_TRUTH_WORDS}
word_errors_english = {w: 0 for w in GROUND_TRUTH_WORDS}
word_counts_russian = {w: 0 for w in GROUND_TRUTH_WORDS}
word_counts_english = {w: 0 for w in GROUND_TRUTH_WORDS}

with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    current_speaker = None
    
    std_wer = 0.0
    orc_wer = 0.0
    ref_list = []
    mark_list = []
    
    for row in reader:
        if not row:
            continue
        speaker = row[0]
        row_type = row[1]
        
        lang = "Russian" if "russian" in speaker else "English"
        
        if row_type == "WER":
            std_wer = float(row[2].split(":")[1].replace("%", "").strip())
            orc_wer = float(row[3].split(":")[1].replace("%", "").strip())
        elif row_type == "Truth":
            ref_list = row[2:]
        elif row_type == "Eval":
            mark_list = row[2:]
            
            # Save speaker metrics
            speakers_data.append({
                "Speaker": speaker,
                "Language": lang,
                "Standard_WER": std_wer,
                "Oracle_WER": orc_wer
            })
            
            # Map word-level evaluation
            ref_word_idx = 0
            for ref_w, mark in zip(ref_list, mark_list):
                if ref_w != "***":
                    if ref_word_idx < len(GROUND_TRUTH_WORDS):
                        word = GROUND_TRUTH_WORDS[ref_word_idx]
                        is_error = 1 if mark == "X" else 0
                        
                        if lang == "Russian":
                            word_errors_russian[word] += is_error
                            word_counts_russian[word] += 1
                        else:
                            word_errors_english[word] += is_error
                            word_counts_english[word] += 1
                            
                        ref_word_idx += 1

# Convert speaker data to Pandas DataFrame
df = pd.DataFrame(speakers_data)

# Compute word-level failure rates
russian_word_stats = []
english_word_stats = []

for i, word in enumerate(GROUND_TRUTH_WORDS):
    err_ru = word_errors_russian[word]
    cnt_ru = word_counts_russian[word]
    rate_ru = (err_ru / cnt_ru * 100) if cnt_ru > 0 else 0
    
    err_en = word_errors_english[word]
    cnt_en = word_counts_english[word]
    rate_en = (err_en / cnt_en * 100) if cnt_en > 0 else 0
    
    # Store word metrics with original index to disambiguate identical words
    russian_word_stats.append({"Word": f"{word} (#{i+1})", "WordRaw": word, "Error_Rate": rate_ru, "Language": "Russian"})
    english_word_stats.append({"Word": f"{word} (#{i+1})", "WordRaw": word, "Error_Rate": rate_en, "Language": "English"})

df_words_ru = pd.DataFrame(russian_word_stats)
df_words_en = pd.DataFrame(english_word_stats)

# Select the top 10 most problematic words for each group
top_ru_words = df_words_ru.sort_values(by="Error_Rate", ascending=False).head(10)
top_en_words = df_words_en.sort_values(by="Error_Rate", ascending=False).head(10)

print("\n=== GLOBAL STATISTICS SUMMARY ===")
print(df.groupby("Language")[["Standard_WER", "Oracle_WER"]].mean())

# --- Plot Visualizations ---
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

# Premium color palette: deep blue and coral/teal
colors = {"Russian": "#E76F51", "English": "#2A9D8F"}

# ----------------------------------------------------
# Plot 1: WER Distributions (Standard vs. Oracle Boxplot)
# ----------------------------------------------------
plt.figure(figsize=(10, 6.5))

df_melted = pd.melt(df, id_vars=["Speaker", "Language"], 
                    value_vars=["Standard_WER", "Oracle_WER"], 
                    var_name="Metric", value_name="WER")
df_melted["Metric"] = df_melted["Metric"].map({"Standard_WER": "Standard WER (Greedy)", "Oracle_WER": "Oracle WER (Best-of-3)"})

ax1 = sns.boxplot(x="Language", y="WER", hue="Metric", data=df_melted,
                  palette=["#1D3557", "#2A9D8F"], width=0.6, linewidth=1.5, fliersize=4)

plt.title("ASR Word Error Rate (WER) Distribution\nRussian vs. English Speakers (Standard vs. Oracle)", pad=15, weight='bold')
plt.xlabel("Speaker Accent Group", labelpad=10, weight='bold')
plt.ylabel("Word Error Rate (WER %)", labelpad=10, weight='bold')
plt.ylim(0, 100)
plt.legend(frameon=True, facecolor="white", edgecolor="none")
plt.tight_layout()

plot1_path = os.path.join(PLOTS_DIR, "wer_comparison.png")
plt.savefig(plot1_path, dpi=300)
if os.path.exists(ARTIFACTS_DIR):
    plt.savefig(os.path.join(ARTIFACTS_DIR, "wer_comparison.png"), dpi=300)
plt.close()

# ----------------------------------------------------
# Plot 2: Top 10 Most Problematic Words Barplots
# ----------------------------------------------------
fig, (ax_ru, ax_en) = plt.subplots(1, 2, figsize=(15, 7))

sns.barplot(x="Error_Rate", y="Word", data=top_ru_words, ax=ax_ru, color="#E76F51", edgecolor="none")
ax_ru.set_title("Top 10 Most Problematic Words\nfor Russian Speakers", pad=12, weight='bold')
ax_ru.set_xlabel("Word Error Rate (WER %)", weight='bold')
ax_ru.set_ylabel("Word (with baseline index #)", weight='bold')
ax_ru.set_xlim(0, 100)

sns.barplot(x="Error_Rate", y="Word", data=top_en_words, ax=ax_en, color="#2A9D8F", edgecolor="none")
ax_en.set_title("Top 10 Most Problematic Words\nfor Native English Speakers", pad=12, weight='bold')
ax_en.set_xlabel("Word Error Rate (WER %)", weight='bold')
ax_en.set_ylabel("", weight='bold')
ax_en.set_xlim(0, 100)

plt.suptitle("ASR Word-Level Error Analysis - Highest Failure Rates", y=0.98, weight='bold')
plt.tight_layout()

plot2_path = os.path.join(PLOTS_DIR, "problematic_words.png")
plt.savefig(plot2_path, dpi=300)
if os.path.exists(ARTIFACTS_DIR):
    plt.savefig(os.path.join(ARTIFACTS_DIR, "problematic_words.png"), dpi=300)
plt.close()

print(f"\n[+] Visualizations successfully saved!")
print(f"  -> WER Boxplot: {plot1_path}")
print(f"  -> Problematic Words Chart: {plot2_path}")
