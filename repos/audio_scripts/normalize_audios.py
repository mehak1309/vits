import os
from glob import glob
from pydub import AudioSegment, effects
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import concurrent


# List of language directories
languages = [
    "Assamese", "Bengali", "Bodo", "Dogri", "Gujarati", "Hindi", "Kannada", "Kashmiri",
    "Konkani", "Maithili", "Malayalam", "Manipuri", "Marathi", "Nepali", "Odia", 
    "Punjabi", "Sanskrit", "Santali", "Sindhi", "Tamil", "Telugu", "Urdu"
]

# Gather all wav files from the wavs directory of each language
files = []
for language in languages:
    files.extend(glob(f"/home/tts/ttsteam/datasets/indicvoices_r/{language}/wavs/*.wav"))
print(f'{len(files)} files found!')


# for file in tqdm(files):
def normalize_audio(file):
    rawsound = AudioSegment.from_file(file, "wav")  
    normalizedsound = effects.normalize(rawsound)  
    normalizedsound.export(file, format="wav")

with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(normalize_audio, file) for file in files]
    done = 0
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        # if done % 100 == 0:
            # print(done, len(files))
        done += 1
