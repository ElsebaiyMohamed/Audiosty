import librosa


audio_path = r"D:\Projects\Audio_funct\uploads\OSR_us_000_0011_8k.wav"
audio, sr = librosa.load(audio_path)
ons = librosa.onset.onset_detect(y=audio, sr=sr, units='samples')

print(librosa.feature.tempo(y=audio, sr=sr, onset_envelope=ons,))