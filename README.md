# voice-based-biometric-system
This project verifies a speaker's identity by comparing two voice samples using MFCC features and cosine similarity. If the similarity exceeds a set threshold, the speaker is authenticated. It's a lightweight, effective voice-based biometric verification system.
#  Install dependencies
!apt-get install ffmpeg -y > /dev/null
!pip install numpy scipy matplotlib soundfile pydub librosa --quiet

from google.colab import files
import librosa
import numpy as np
import soundfile as sf
import os

# 📁 Upload audio samples
print("Upload the ENROLLED voice sample:")
enrolled_file = files.upload()

print("Upload the TEST voice sample:")
test_file = files.upload()

# Convert to WAV if needed
from pydub import AudioSegment
def convert_to_wav(filepath):
    if filepath.endswith(".wav"):
        return filepath
    audio = AudioSegment.from_file(filepath)
    wav_path = filepath.rsplit('.', 1)[0] + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path

enroll_path = convert_to_wav(list(enrolled_file.keys())[0])
test_path = convert_to_wav(list(test_file.keys())[0])

# Load audio and extract MFCC
def extract_mfcc(filepath):
    y, sr = librosa.load(filepath, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)  # Mean MFCC vector

enroll_mfcc = extract_mfcc(enroll_path)
test_mfcc = extract_mfcc(test_path)

# Cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

similarity = cosine_similarity(enroll_mfcc, test_mfcc)
print(f"\n🔍 Cosine Similarity: {similarity:.3f}")

# ✅ Threshold check
threshold = 0.85
if similarity >= threshold:
    print("✅ Speaker Verified")
else:
    print("❌ Speaker Not Verified")
