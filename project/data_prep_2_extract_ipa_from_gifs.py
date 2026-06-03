import os
import time
import re
from PIL import Image
import google.genai as genai

# --- Spoken IPA Text Extractor using Gemini API ---
# Loads the downloaded phonetic transcription image GIF from each speaker folder,
# sends it to the Gemini Multimodal API to extract the raw IPA characters,
# cleans diacritics/stress markers to establish broad IPA transcripts,
# and saves it as speaker_ipa.txt inside the speaker's data folder.

GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=GOOGLE_API_KEY)

prompt = """
Extract the exact phonetic transcription from this image. 
Keep all IPA characters and formatting intact. 
Return ONLY the raw phonetic text inside the brackets, without any conversational filler or markdown formatting.
"""

def clean_ipa_to_broad(raw_text):
    # Removes formatting brackets, stress marks, length markers, and non-broad diacritics
    text = re.sub(r'[\[\]]', '', raw_text)
    text = re.sub(r'[ʰː:ˈˌ̚ⁿˡʲʷ]', '', text)
    text = re.sub(r'[\u0300-\u036f]', '', text) # Removes unicode combining diacritical marks
    text = re.sub(r'\s+', ' ', text).strip()
    return text

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(script_dir, "data", "accent_dataset")
os.makedirs(dataset_dir, exist_ok=True)

print("Starting the IPA extraction and cleaning process...")

for speaker_folder in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, speaker_folder)
    
    if not os.path.isdir(folder_path):
        continue

    # Filter for english1 to english82 and russian1 to russian82
    match = re.match(r'^(english|russian)(\d+)$', speaker_folder)
    if not match or not (1 <= int(match.group(2)) <= 82):
        continue

    # Paths for the GIF and the output text file
    gif_path = os.path.join(folder_path, f"{speaker_folder}_ipa.gif")
    txt_output_path = os.path.join(folder_path, f"{speaker_folder}_ipa.txt")
    
    if not os.path.exists(gif_path):
        continue
    if os.path.exists(txt_output_path):
        print(f"Already processed {speaker_folder}")
        continue

    print(f"Processing {speaker_folder}...")
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Load the image using PIL
            img = Image.open(gif_path)
            
            # Send prompt and image to gemini-3.1-flash-lite for fast API response
            response = client.models.generate_content(
                model='gemini-3.1-flash-lite',
                contents=[prompt, img]
            )
            
            raw_ipa_text = response.text.strip()
            broad_ipa_text = clean_ipa_to_broad(raw_ipa_text)
            
            with open(txt_output_path, 'w', encoding='utf-8') as f:
                f.write(broad_ipa_text)
                
            print(f"  -> Extracted & Cleaned: {broad_ipa_text[:30]}...") 
            break
            
        except Exception as e:
            error_str = str(e)
            if '429' in error_str or '503' in error_str:
                print(f"  -> API overloaded (429/503), waiting 60s before retry (attempt {attempt+1}/{max_retries})...")
                time.sleep(60)
            else:
                print(f"  [X] Error in processing {speaker_folder}: {e}")
                break

    # Avoid hitting API rate limits too quickly
    time.sleep(8)

print("\n--- The Spoken IPA Extraction process is complete ---")
