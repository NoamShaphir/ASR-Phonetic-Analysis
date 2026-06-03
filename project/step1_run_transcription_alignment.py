import os
import torch
import librosa
import re
import csv
import time
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# --- Step 1: Run ASR Transcription & Aligned Report Generator ---
# Runs Wav2Vec2 locally on CUDA (GPU) if available, processing all 164 speakers.
# Performs CTC Beam Search to extract the top 3 word options with confidence metrics.
# Automatically aligns target standard text, spoken IPA, evaluations, and predictions
# character-by-character (word-by-word) using a Consonant-Skeleton Levenshtein alignment.
# Outputs a structured, highly formatted report to project/outputs/aligned_transcription_report.csv

script_dir = os.path.dirname(os.path.abspath(__file__))
LOCAL_DIR = os.path.join(script_dir, "wav2vec2-base-local")
BASE_DATASET_DIR = os.path.join(script_dir, "data", "accent_dataset")
OUTPUT_CSV_PATH = os.path.join(script_dir, "outputs", "aligned_transcription_report.csv")

os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)

# Ground Truth Raw Elicitation Paragraph
GROUND_TRUTH_RAW = (
    "Please call Stella.  Ask her to bring these things with her from the store:  "
    "Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.  "
    "We also need a small plastic snake and a big toy frog for the kids.  "
    "She can scoop these things into three red bags, and we will go meet her Wednesday at the train station."
)

# Text cleaning for models (only uppercase English letters and apostrophes)
GROUND_TRUTH_CLEAN = re.sub(r"[^A-Z'\s]", " ", GROUND_TRUTH_RAW.upper())
GROUND_TRUTH_WORDS = GROUND_TRUTH_CLEAN.split()

# Extract consonant skeleton to facilitate precise cross-lingual phonetic matching
def consonant_skeleton(word, is_ipa=False):
    word = word.lower()
    if is_ipa:
        # Mapping IPA phone variations back to familiar English consonants for alignment
        ipa_map = {
            'ɡ': 'g', 'ŋ': 'n', 'ɹ': 'r', 'r': 'r', 'ɾ': 'r', 'ʃ': 's', 
            'ʒ': 's', 'θ': 't', 'ð': 't', 'ʋ': 'v', 'ɫ': 'l', 'c': 'k',
            'x': 'h', 'h': 'h', 'ʔ': ''
        }
        chars = []
        for c in word:
            c = ipa_map.get(c, c)
            if c in 'bdfghjklmnpqrstvwxz':
                chars.append(c)
        return "".join(chars)
    else:
        chars = []
        for c in word:
            if c == 'c':
                chars.append('k')
            elif c in 'bdfghjklmnpqrstvwxz':
                chars.append(c)
        return "".join(chars)

# Levenshtein Alignment of Spoken IPA to Ground Truth English words using Consonant Skeletons
def get_aligned_ipa_levenshtein(ref_words, ipa_text):
    ipa_words = ipa_text.split()
    n, m = len(ref_words), len(ipa_words)
    if m == 0:
        return ["***"] * n
        
    ref_skeletons = [consonant_skeleton(w, is_ipa=False) for w in ref_words]
    ipa_skeletons = [consonant_skeleton(w, is_ipa=True) for w in ipa_words]
    
    # DP Table
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i * 1.0
    for j in range(m + 1): dp[0][j] = j * 1.0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            ref_sk = ref_skeletons[i-1]
            ipa_sk = ipa_skeletons[j-1]
            
            if ref_sk == ipa_sk:
                cost = 0.0
            elif (ref_sk and ipa_sk) and (ref_sk in ipa_sk or ipa_sk in ref_sk):
                cost = 0.2  # Highly similar overlap
            else:
                cost = 0.8 if (ref_sk and ipa_sk) else 1.0
                
            dp[i][j] = min(dp[i-1][j] + 1.0,
                           dp[i][j-1] + 1.0,
                           dp[i-1][j-1] + cost)
                            
    # DP Backtracking
    i, j = n, m
    aligned_ipa = []
    
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            ref_sk = ref_skeletons[i-1]
            ipa_sk = ipa_skeletons[j-1]
            cost = 0.0 if ref_sk == ipa_sk else (0.2 if (ref_sk and ipa_sk and (ref_sk in ipa_sk or ipa_sk in ref_sk)) else (0.8 if (ref_sk and ipa_sk) else 1.0))
            if abs(dp[i][j] - (dp[i-1][j-1] + cost)) < 1e-4:
                aligned_ipa.append((i-1, ipa_words[j-1]))
                i -= 1; j -= 1
                continue
                
        if i > 0 and abs(dp[i][j] - (dp[i-1][j] + 1.0)) < 1e-4:
            aligned_ipa.append((i-1, "***"))
            i -= 1
        else:
            aligned_ipa.append((-1, ipa_words[j-1]))
            j -= 1
            
    aligned_ipa.reverse()
    
    # Map back to a ref_words list, merging split words if needed
    result = ["***"] * n
    last_valid_ref_idx = 0
    
    for ref_idx, ipa_word in aligned_ipa:
        if ref_idx != -1:
            if result[ref_idx] == "***":
                result[ref_idx] = ipa_word
            else:
                result[ref_idx] += " " + ipa_word
            last_valid_ref_idx = ref_idx
        else:
            if result[last_valid_ref_idx] == "***":
                result[last_valid_ref_idx] = ipa_word
            else:
                result[last_valid_ref_idx] += " " + ipa_word
                
    return result

# Levenshtein word distance and aligned index generator
def get_aligned_lists(ref_words, hyp_words):
    n, m = len(ref_words), len(hyp_words)
    
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,
                           dp[i][j-1] + 1,
                           dp[i-1][j-1] + cost)

    i, j = n, m
    align_ref, align_hyp, align_mark = [], [], []
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] and ref_words[i-1] == hyp_words[j-1]:
            align_ref.append(ref_words[i-1])
            align_hyp.append(hyp_words[j-1])
            align_mark.append('V')
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            align_ref.append(ref_words[i-1])
            align_hyp.append(hyp_words[j-1])
            align_mark.append('X')
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            align_ref.append(ref_words[i-1])
            align_hyp.append("***")
            align_mark.append('X')
            i -= 1
        else:
            align_ref.append("***")
            align_hyp.append(hyp_words[j-1])
            align_mark.append('X')
            j -= 1
            
    align_ref.reverse()
    align_hyp.reverse()
    align_mark.reverse()
    
    wer = (dp[n][m] / n) * 100 if n > 0 else 0
    return align_ref, align_mark, align_hyp, wer

# Calculates Oracle WER (taking the best out of 3 Beam Search candidates for each word segment)
def get_oracle_wer(ref_words, segment_options):
    n, m = len(ref_words), len(segment_options)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            opts = segment_options[j-1]
            cost = 0 if ref_words[i-1] in opts else 1
            dp[i][j] = min(dp[i-1][j] + 1,
                           dp[i][j-1] + 1,
                           dp[i-1][j-1] + cost)
    return (dp[n][m] / n) * 100 if n > 0 else 0

# CTC Beam Search decoder for individual greedy word segments
def ctc_beam_search_for_word(probs_chunk, blank_id, space_id, id_to_char, beam_size=10):
    paths = {(): [1.0, 0.0]} 
    
    for t in range(probs_chunk.shape[0]):
        new_paths = {}
        prob_t = probs_chunk[t]
        
        top_k_probs, top_k_ids = torch.topk(prob_t, min(5, prob_t.shape[0]))
        
        for seq, (p_b, p_nb) in paths.items():
            p_blank_ext = (p_b + p_nb) * prob_t[blank_id].item()
            if seq not in new_paths:
                new_paths[seq] = [0.0, 0.0]
            new_paths[seq][0] += p_blank_ext
            
            for i in range(len(top_k_ids)):
                v = top_k_ids[i].item()
                prob_v = top_k_probs[i].item()
                
                if v == blank_id or v == space_id:
                    continue
                    
                if prob_v < 1e-4:
                    continue
                    
                if len(seq) > 0 and v == seq[-1]:
                    p_nb_ext_a = p_b * prob_v
                    new_seq_a = seq + (v,)
                    if new_seq_a not in new_paths:
                        new_paths[new_seq_a] = [0.0, 0.0]
                    new_paths[new_seq_a][1] += p_nb_ext_a
                    
                    p_nb_ext_b = p_nb * prob_v
                    if seq not in new_paths:
                        new_paths[seq] = [0.0, 0.0]
                    new_paths[seq][1] += p_nb_ext_b
                else:
                    p_nb_ext = (p_b + p_nb) * prob_v
                    new_seq = seq + (v,)
                    if new_seq not in new_paths:
                        new_paths[new_seq] = [0.0, 0.0]
                    new_paths[new_seq][1] += p_nb_ext
                    
        sorted_paths = sorted(new_paths.items(), key=lambda x: x[1][0] + x[1][1], reverse=True)[:beam_size]
        total_prob = sum(p[0] + p[1] for _, p in sorted_paths)
        if total_prob > 0:
            paths = {seq: [p[0]/total_prob, p[1]/total_prob] for seq, p in sorted_paths}
        else:
            paths = {seq: p for seq, p in sorted_paths}
            
    res = []
    for seq, (p_b, p_nb) in paths.items():
        word_str = "".join([id_to_char.get(i, "") for i in seq])
        if word_str.strip(): 
            res.append((word_str, p_b + p_nb))
            
    final_dict = {}
    for w, p in res:
        final_dict[w] = final_dict.get(w, 0.0) + p
        
    final_res = sorted(final_dict.items(), key=lambda x: x[1], reverse=True)
    return final_res

# --- Pipeline Execution ---
print("Loading model and processor...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device selected: {device}")

processor = Wav2Vec2Processor.from_pretrained(LOCAL_DIR)
model = Wav2Vec2ForCTC.from_pretrained(LOCAL_DIR).to(device)
print("Model loaded locally successfully!")

vocab_dict = processor.tokenizer.get_vocab()
id_to_char = {v: k for k, v in vocab_dict.items()}
space_id = vocab_dict.get("|", processor.tokenizer.word_delimiter_token_id)
blank_id = processor.tokenizer.pad_token_id

speakers = []
for i in range(1, 83): speakers.append(f"russian{i}")
for i in range(1, 83): speakers.append(f"english{i}")

print(f"Total speakers detected: {len(speakers)}")
print(f"Executing greedy/beam search inference and aligning output CSV...")

start_time = time.time()
success_count = 0

with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8-sig") as csv_file:
    writer = csv.writer(csv_file)
    
    for idx, speaker_name in enumerate(speakers, 1):
        audio_path = os.path.join(BASE_DATASET_DIR, speaker_name, f"{speaker_name}.mp3")
        ipa_path = os.path.join(BASE_DATASET_DIR, speaker_name, f"{speaker_name}_ipa.txt")
        
        if not os.path.exists(audio_path):
            print(f"[{idx}/{len(speakers)}] Skip: {speaker_name} (Audio not found)")
            continue
            
        print(f"[{idx}/{len(speakers)}] Transcribing & Aligning: {speaker_name}...")
        
        try:
            # 1. Load Audio and resample
            speech_array, sr = librosa.load(audio_path, sr=16000)
            inputs = processor(speech_array, sampling_rate=16000, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
            
            # 2. Grab probabilities and greedy segments
            probabilities = torch.nn.functional.softmax(logits[0], dim=-1).cpu()
            best_path = torch.argmax(probabilities, dim=-1)
            
            word_segments = []
            start_idx = 0
            for i, token_id in enumerate(best_path):
                if token_id.item() == space_id:
                    if start_idx < i:
                        word_segments.append((start_idx, i - 1))
                    start_idx = i + 1
            if start_idx < len(best_path):
                word_segments.append((start_idx, len(best_path) - 1))
                
            # 3. Beam Search decoded alternatives for each segment
            segment_options_list = []
            options_metadata = []
            greedy_hyp_words = []
            
            for start, end in word_segments:
                if end < start:
                    continue
                chunk_probs = probabilities[start:end+1]
                top_words = ctc_beam_search_for_word(chunk_probs, blank_id, space_id, id_to_char, beam_size=10)
                
                opts = []
                meta = []
                for j in range(3):
                    if j < len(top_words):
                        w, p = top_words[j]
                        opts.append(w.upper())
                        meta.append((w.upper(), p))
                    else:
                        opts.append("")
                        meta.append(("", 0.0))
                
                segment_options_list.append(opts)
                options_metadata.append(meta)
                greedy_hyp_words.append(opts[0])
                
            # 4. Spoken IPA Levenshtein Alignment using Consonant Skeletons
            if os.path.exists(ipa_path):
                with open(ipa_path, "r", encoding="utf-8") as f:
                    ipa_text = f.read().strip()
                ipa_aligned = get_aligned_ipa_levenshtein(GROUND_TRUTH_WORDS, ipa_text)
            else:
                # English reference broad IPA transcription fallback (for unannotated GMU Russian speakers)
                ipa_aligned = [
                    "pliz", "kɑl", "stɛlʌ", "æsk", "hɚ", "tu", "bɹɪŋ", "ðiz", "θɪŋz", "wɪθ", 
                    "hɚ", "fɹʌm", "ðʌ", "stɔɹ", "sɪks", "spunz", "ʌv", "fɹɛʃ", "snoʊ", "piz", 
                    "faɪv", "θɪk", "slæbz", "ʌv", "blu", "tʃiz", "ænd", "meɪbi", "ʌ", "snæk", 
                    "fɔɹ", "hɚ", "bɹʌðɚ", "bɑb", "wi", "ɑlsoʊ", "nid", "ʌ", "smɑl", "plæstɪk", 
                    "sneɪk", "ænd", "ʌ", "bɪɡ", "tɔɪ", "fɹɑɡ", "fɔɹ", "ðʌ", "kɪdz", "ʃi", 
                    "kæn", "skup", "ðiz", "θɪŋz", "ɪntu", "θɹi", "ɹɛd", "bæɡz", "ænd", "wi", 
                    "wɪl", "goʊ", "mit", "hɚ", "wɛnzdeɪ", "æt", "ðʌ", "tɹeɪn", "steɪʃʌn"
                ]
                
            # 5. Greedy alignment
            ref_list, mark_list, hyp_list, wer_score = get_aligned_lists(GROUND_TRUTH_WORDS, greedy_hyp_words)
            
            # 6. Oracle alignment
            oracle_wer_score = get_oracle_wer(GROUND_TRUTH_WORDS, segment_options_list)
            
            # 7. Rearrange alignment rows
            aligned_ipa_row = []
            aligned_opt1_row = []
            aligned_opt2_row = []
            aligned_opt3_row = []
            
            j = 0
            for k in range(len(hyp_list)):
                if hyp_list[k] == "***":
                    aligned_opt1_row.append("***")
                    aligned_opt2_row.append("***")
                    aligned_opt3_row.append("***")
                else:
                    meta = options_metadata[j]
                    
                    w1, p1 = meta[0]
                    aligned_opt1_row.append(f"{w1} ({p1*100:.1f}%)" if w1 else "***")
                    
                    w2, p2 = meta[1]
                    aligned_opt2_row.append(f"{w2} ({p2*100:.1f}%)" if w2 else "")
                    
                    w3, p3 = meta[2]
                    aligned_opt3_row.append(f"{w3} ({p3*100:.1f}%)" if w3 else "")
                    
                    j += 1
            
            ipa_ptr = 0
            for k in range(len(ref_list)):
                if ref_list[k] == "***":
                    aligned_ipa_row.append("***")
                else:
                    aligned_ipa_row.append(ipa_aligned[ipa_ptr])
                    ipa_ptr += 1
                    
            # 8. Write speaker block to CSV
            row_wer   = [speaker_name, "WER", f"Standard: {wer_score:.2f}%", f"Oracle: {oracle_wer_score:.2f}%"]
            row_truth = [speaker_name, "Truth"] + ref_list
            row_ipa   = [speaker_name, "IPA"] + aligned_ipa_row
            row_mark  = [speaker_name, "Eval"] + mark_list
            row_opt1  = [speaker_name, "Pred_Opt1"] + aligned_opt1_row
            row_opt2  = [speaker_name, "Pred_Opt2"] + aligned_opt2_row
            row_opt3  = [speaker_name, "Pred_Opt3"] + aligned_opt3_row
            
            writer.writerow(row_wer)
            writer.writerow(row_truth)
            writer.writerow(row_ipa)
            writer.writerow(row_mark)
            writer.writerow(row_opt1)
            writer.writerow(row_opt2)
            writer.writerow(row_opt3)
            writer.writerow([])
            
            success_count += 1
            
        except Exception as e:
            print(f"  [X] Error processing {speaker_name}: {e}")

end_time = time.time()
print(f"\n[+] Processing complete! Clean aligned report saved to: {OUTPUT_CSV_PATH}")
print(f"Total time elapsed: {end_time - start_time:.2f} seconds.")
