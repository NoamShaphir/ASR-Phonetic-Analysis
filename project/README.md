# 🎙️ GMU Speech Accent Archive ASR & Phonetic Analysis Pipeline

Developer guide and execution manual for the **Russian vs. Native English ASR Evaluation and Phonetic Ambiguity Deep-Dive** pipeline.

---

## 📂 Project Directory Structure

```
project/
├── data/
│   ├── accent_dataset/         # Speaker audio files & extracted text broad IPAs (.mp3, _ipa.txt)
│   └── drive-download-.../     # User-supplied new test dataset of WAV audios & TXT transcripts
├── docs/                       # Comprehensive documentation & research reports
│   ├── ASR_Phonetic_Analysis_Master_Report.md   # Unified master report in Markdown
│   ├── ASR_Phonetic_Analysis_Master_Report.docx # Unified master report in MS Word format
│   ├── New_Test_Dataset_Evaluation_Report.md    # Unseen test dataset evaluation in Markdown
│   └── New_Test_Dataset_Evaluation_Report.docx  # Unseen test dataset evaluation in MS Word format
├── outputs/                    # Processed CSV reports and text reports
│   ├── aligned_transcription_report.csv   # Aligned CSV (WER, Truth, IPA, Eval, 3 Beam Candidates)
│   ├── raw_model_transcriptions.csv      # Unaligned raw word confidence predictions from Wav2Vec2
│   ├── phonetic_errors.txt               # Global ranking of phonetic confusions (spoken -> predicted)
│   ├── phonetic_pattern_results.txt       # Exact phonetic counts and voicing/manner shifts on target words
│   └── new_test_dataset_evaluation_report.txt # Text log of unseen test dataset results
├── plots/                      # Premium visualizations and analytical heatmaps
│   ├── wer_comparison.png                # Boxplot comparing Standard vs. Oracle WER scores
│   ├── problematic_words.png             # Barplot of the top 10 most problematic words per group
│   ├── confusion_heatmap_ru/en.png       # 26x26 full A-Z orthographic character confusion heatmaps
│   ├── error_heatmap_ru/en.png           # 26x26 orthographic error-only heatmaps
│   ├── phonetic_heatmap_ru/en.png        # 23x23 full IPA-to-IPA sound confusion heatmaps
│   └── phonetic_error_heatmap_ru/en.png  # 23x23 phonetic error-only heatmaps
├── wav2vec2-base-local/        # Local offline copy of Wav2Vec2 processor and model weights
├── wav2vec2-lora-russian-adapter.pt # Trained PEFT self-attention adapter weights
│
# Data Preparation Utilities
├── data_prep_1_download_speech_archive.py  # Scrapes audio and transcript images from GMU website
├── data_prep_2_extract_ipa_from_gifs.py    # Extracts clean broad IPA from GMU GIFs using Gemini API
├── data_prep_3_download_wav2vec2_model.py  # Downloads facebook/wav2vec2-base-960h weights locally
│
# Core Execution Pipeline Steps
├── step1_run_transcription_alignment.py    # GPU-accelerated ASR transcription and Levenshtein alignment
├── step2_analyze_wer.py                    # Evaluates WER, standard vs. oracle, plots boxplots
├── step3_analyze_phonetic_confusion.py     # Phoneme G2P decoder, 23x23 heatmaps & phonetic error logger
├── step4_analyze_orthographic_confusion.py # Character Levenshtein alignment & 26x26 A-Z heatmaps
├── step5_analyze_target_words.py           # Deep accent analysis on slabs, these, things, kids, thick
├── step6_train_lora_adaptation.py          # PEFT/LoRA adapter training blueprint (q_proj & v_proj unfreezing)
├── evaluate_new_data.py                    # Runs evaluation on unseen test dataset using baseline vs. LoRA adapter
├── analyze_duration.py                    # Performs robust, IQR-cleaned vowel duration analysis
├── compare_manual_vs_model_ipa.py          # Aligns manual IPA from researchers with model G2P predicted IPA
│
# Helper Scripts
└── scratch/
    ├── compile_docx.py                     # Compiles MD files to custom Word document styling
    ├── calculate_ipa_mismatch_rate.py     # Computes manual vs model G2P mismatch rates (russian10 vs global)
    └── evaluate_catastrophic_forgetting.py # Runs validation checks on Native English speakers after adaptation
```

---

## 🛠️ Prerequisites & Installation

Ensure you have **Python 3.10** installed. It is highly recommended to run this inside a virtual environment.

1. **Activate the Virtual Environment:**
   ```powershell
   # Windows PowerShell
   .\data\.venv\Scripts\activate
   ```
2. **Install Required Packages:**
   Ensure you have all necessary deep learning, signal processing, and charting libraries installed:
   ```bash
   pip install torch transformers librosa eng-to-ipa pandas numpy matplotlib seaborn requests pillow google-genai python-docx
   ```

---

## 🚀 Data Preparation Scripts

If you need to re-download the dataset or download weights locally:

1. **Download GMU Audio and GIFs:**
   ```bash
   python data_prep_1_download_speech_archive.py
   ```
2. **Extract Spoken IPA using Gemini:**
   Extracts broad IPA text from local speaker GIF files using the Gemini multimodal API:
   ```bash
   python data_prep_2_extract_ipa_from_gifs.py
   ```
3. **Save Wav2Vec2 Locally for Offline Functionality:**
   ```bash
   python data_prep_3_download_wav2vec2_model.py
   ```

---

## 📊 Pipeline Step-by-Step Execution

Execute the following steps in sequence to run the entire pipeline:

### Step 1: Run ASR Inference and Alignment
Runs GPU-accelerated Wav2Vec2 inference on all 164 audio recordings. Performs CTC beam search decoding for the top 3 candidates and aligns standard words, spoken IPA, evaluations, and predictions to save a rigid CSV block table.
```bash
python step1_run_transcription_alignment.py
```
* **Output:** `outputs/aligned_transcription_report.csv`

### Step 2: Global ASR and WER Boxplot Analysis
Parses the aligned CSV report, calculates standard vs. oracle WER, determines word-level error rates for the elicitation text, and generates comparison boxplots.
```bash
python step2_analyze_wer.py
```
* **Outputs:** `plots/wer_comparison.png`, `plots/problematic_words.png`

### Step 3: 23x23 IPA-to-IPA Confusion Matrices
Decodes predicted ASR graphemes to IPA using G2P and performs character-level phonemic Levenshtein alignment. Generates normalized 23x23 IPA-to-IPA confusion and error heatmaps and writes the rank-ordered phonetic substitutions.
```bash
python step3_analyze_phonetic_confusion.py
```
* **Outputs:** `plots/phonetic_heatmap_ru/en.png`, `plots/phonetic_error_heatmap_ru/en.png`, `outputs/phonetic_errors.txt`

### Step 4: 26x26 Orthographic A-Z Heatmaps
Aligns predicted orthographic words character-by-character to generate standard 26x26 A-Z character confusion and error heatmaps.
```bash
python step4_analyze_orthographic_confusion.py
```
* **Outputs:** `plots/confusion_heatmap_ru/en.png`, `plots/error_heatmap_ru/en.png`

### Step 5: Target Word Russian Accent Deep-Dive
Runs a detailed count of Russian phonological transfers (Final-Consonant Devoicing, th-stopping, sibilant shifts, tense-lax vowel raising) on the 5 target words: `SLABS`, `THESE`, `THINGS`, `KIDS`, and `THICK`.
```bash
python step5_analyze_target_words.py
```
* **Output:** `outputs/phonetic_pattern_results.txt`

### Step 6: Parameter-Efficient Fine-Tuning (PEFT/LoRA)
Fine-tunes the query and value projection matrices (`q_proj`, `v_proj`) in Wav2Vec2's self-attention heads on the Russian speakers to calibrate Mel-spectrogram feature representations.
```bash
python step6_train_lora_adaptation.py
```
* **Output:** `wav2vec2-lora-russian-adapter.pt`

### Step 7: Evaluate Unseen Test Dataset
Transcribes the 5 custom WAV audio files using both the baseline and LoRA-adapted models to evaluate phonetic template performance and word fusion CTC spacing bounds.
```bash
python evaluate_new_data.py
```
* **Outputs:** `docs/New_Test_Dataset_Evaluation_Report.md`, `docs/New_Test_Dataset_Evaluation_Report.docx`, `outputs/new_test_dataset_evaluation_report.txt`

---

## 💡 Phonetic Key Findings

- **ASR Disparity:** Standard WER for Russian speakers is **27.34%**, compared to just **9.51%** for Native English speakers. A relative error reduction of **19.5%** is achieved via Oracle WER (**21.99%**), showing enormous potential for Language Model integration.
- **Final Consonant Devoicing:** Voiced final consonants are devoiced in Russian speakers (**21.7%** of target word errors), transforming `/b/` $\rightarrow$ `/p/` in `SLABS` (*slaps*) and `/d/` $\rightarrow$ `/t/` + `/z/` $\rightarrow$ `/s/` in `KIDS` (*kits*).
- **Dental Fricative Shifts (th-stopping / th-sibilant):** The continuous voiceless dental fricative `/θ/` is replaced by the closed alveolar stop `/t/` (th-stopping, e.g. `THICK` $\rightarrow$ *tick*) or shifted to familiar sibilant `/s/` (sibilant shift, e.g. `THICK` $\rightarrow$ *sick*).
- **Vowel Height / Tenseness Shift:** Because Russian lacks the tense-lax vowel distinction, `/i/` (tense close front vowel) and `/ɪ/` (lax near-close front vowel) are merged, yielding symmetric `/i/` $\leftrightarrow$ `/ɪ/` acoustic confusions.
