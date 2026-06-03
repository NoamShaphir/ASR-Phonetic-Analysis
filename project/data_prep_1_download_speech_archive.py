import os
import requests
import time

# --- GMU Speech Accent Archive Scraper and Downloader ---
# Downloads 82 Russian and 82 Native English speaker recordings (.mp3)
# and phonetic transcription images (.gif) to project/data/accent_dataset/

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "data", "accent_dataset")
os.makedirs(output_dir, exist_ok=True)

russian_speakers = [f"russian{i}" for i in range(1, 83)]
english_speakers = [f"english{i}" for i in range(1, 83)]
all_speakers = russian_speakers + english_speakers

english_text = (
    "Please call Stella.  Ask her to bring these things with her from the store:  "
    "Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.  "
    "We also need a small plastic snake and a big toy frog for the kids.  "
    "She can scoop these things into three red bags, and we will go meet her Wednesday at the train station."
)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

print(f"Dataset target folder: {output_dir}")
print("Starting download process for 164 speakers...")

for speaker in all_speakers:
    speaker_dir = os.path.join(output_dir, speaker)
    os.makedirs(speaker_dir, exist_ok=True)
    
    # 1. Save English transcript text file
    text_path = os.path.join(speaker_dir, f"{speaker}_transcript.txt")
    if not os.path.exists(text_path):
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(english_text)

    # 2. Download speaker audio file (MP3)
    audio_path = os.path.join(speaker_dir, f"{speaker}.mp3")
    if not os.path.exists(audio_path):
        audio_url = f"https://accent.gmu.edu/soundtracks/{speaker}.mp3"
        try:
            audio_response = requests.get(audio_url, headers=headers, timeout=15)
            if audio_response.status_code == 200:
                with open(audio_path, 'wb') as f:
                    f.write(audio_response.content)
                print(f"  [+] Downloaded Audio: {speaker}.mp3")
            else:
                print(f"  [-] Audio not found for {speaker} (HTTP {audio_response.status_code})")
        except Exception as e:
            print(f"  [X] Network error downloading audio for {speaker}: {e}")

    # 3. Download spoken IPA transcription image file (GIF)
    gif_path = os.path.join(speaker_dir, f"{speaker}_ipa.gif")
    if not os.path.exists(gif_path):
        gif_url = f"https://accent.gmu.edu/ipagifs/{speaker}.gif"
        try:
            gif_response = requests.get(gif_url, headers=headers, timeout=15)
            if gif_response.status_code == 200:
                with open(gif_path, 'wb') as f:
                    f.write(gif_response.content)
                print(f"  [+] Downloaded Transcription GIF: {speaker}_ipa.gif")
            else:
                print(f"  [-] GIF not found for {speaker} (HTTP {gif_response.status_code})")
        except Exception as e:
            print(f"  [X] Network error downloading GIF for {speaker}: {e}")

    # Critical rate limiting delay to avoid IP blocking from GMU academic servers
    time.sleep(1.5)

print("\n--- Speech Accent Archive Scraper process is complete ---")
