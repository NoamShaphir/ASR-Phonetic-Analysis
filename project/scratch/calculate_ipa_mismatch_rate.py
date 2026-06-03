import csv
import os
import sys
import re
import eng_to_ipa as ipa

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Paths
script_dir = r"c:\Users\Noam\OneDrive\Desktop\Noam\year_3\voice\project"
csv_path = os.path.join(script_dir, "outputs", "aligned_transcription_report.csv")
fixed_ipa_dir = r"C:\Users\Noam\OneDrive\Desktop\fixed_IPA"

# Cache for G2P
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

def align_words_levenshtein(ref_list, hyp_list):
    n, m = len(ref_list), len(hyp_list)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_list[i-1] == hyp_list[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,
                           dp[i][j-1] + 1,
                           dp[i-1][j-1] + cost)
                           
    i, j = n, m
    align_ref, align_hyp = [], []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref_list[i-1] == hyp_list[j-1] else 1
            if dp[i][j] == dp[i-1][j-1] + cost:
                align_ref.append(ref_list[i-1])
                align_hyp.append(hyp_list[j-1])
                i -= 1; j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i-1][j] + 1:
            align_ref.append(ref_list[i-1])
            align_hyp.append("***")
            i -= 1
        else:
            align_ref.append("***")
            align_hyp.append(hyp_list[j-1])
            j -= 1
            
    align_ref.reverse()
    align_hyp.reverse()
    return align_ref, align_hyp

def get_speaker_mismatch_rate(speaker_name, rows):
    manual_ipa_words = []
    pred_words = []
    
    # 1. Load manual spoken IPA from fixed_IPA if exists, otherwise fallback to CSV/original
    fixed_file = os.path.join(fixed_ipa_dir, f"{speaker_name}_ipa.txt")
    if os.path.exists(fixed_file):
        with open(fixed_file, "r", encoding="utf-8") as f:
            manual_ipa_text = " ".join([line.strip() for line in f.readlines() if line.strip()])
            manual_ipa_text = re.sub(r'[\[\]]', '', manual_ipa_text)
            manual_ipa_words = [w for w in manual_ipa_text.split() if w.strip()]
    else:
        # Fallback to aligned report CSV
        found = False
        for i in range(len(rows)):
            if rows[i] and rows[i][0] == speaker_name:
                found = True
                for j in range(8):
                    if i + j < len(rows):
                        row = rows[i+j]
                        if len(row) > 1 and row[1] == "IPA":
                            manual_ipa_words = [w for w in row[2:] if w != "***" and w.strip() and w != "[No IPA]"]
                break
        if not found or not manual_ipa_words:
            return None
            
    # Extract model predicted words
    found = False
    for i in range(len(rows)):
        if rows[i] and rows[i][0] == speaker_name:
            found = True
            for j in range(8):
                if i + j < len(rows):
                    row = rows[i+j]
                    if len(row) > 1 and row[1] == "Pred_Opt1":
                        for w in row[2:]:
                            if w == "***" or not w.strip():
                                continue
                            w_clean = w.split(" (")[0]
                            pred_words.append(w_clean)
            break
            
    if not found or not pred_words:
        return None
        
    pred_ipa_words = [word_to_ipa(w) for w in pred_words]
    
    # Align manual spoken IPA to predicted IPA
    aligned_manual, aligned_pred = align_words_levenshtein(manual_ipa_words, pred_ipa_words)
    
    mismatches = 0
    total = len(aligned_manual)
    
    for k in range(total):
        if aligned_manual[k] != aligned_pred[k]:
            mismatches += 1
            
    mismatch_rate = (mismatches / total) * 100 if total > 0 else 0.0
    return mismatch_rate, mismatches, total

def main():
    if not os.path.exists(csv_path):
        print(f"[-] CSV file not found at: {csv_path}")
        sys.exit(1)
        
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = list(reader)
        
    # Calculate for russian10
    ru10_res = get_speaker_mismatch_rate("russian10", rows)
    if ru10_res:
        rate, mismatches, total = ru10_res
        print(f"[russian10] Mismatch Rate (using fixed_IPA): {rate:.2f}% ({mismatches} different words out of {total} total aligned words)")
    else:
        print("[-] Could not calculate for russian10")
        
    # Calculate globally across all 82 Russian speakers
    rates = []
    total_mismatches = 0
    total_words = 0
    
    for i in range(1, 83):
        sp = f"russian{i}"
        res = get_speaker_mismatch_rate(sp, rows)
        if res:
            rate, mismatches, total = res
            rates.append(rate)
            total_mismatches += mismatches
            total_words += total
            
    avg_rate_individual = sum(rates) / len(rates) if rates else 0.0
    overall_rate = (total_mismatches / total_words) * 100 if total_words > 0 else 0.0
    
    print(f"\n[Global Russian Accent Group - 82 Speakers (with fixed_IPA integrations)]")
    print(f"- Average Mismatch Rate per Speaker: {avg_rate_individual:.2f}%")
    print(f"- Overall Global Mismatch Rate: {overall_rate:.2f}% ({total_mismatches} mismatches out of {total_words} total aligned words)")

if __name__ == "__main__":
    main()
