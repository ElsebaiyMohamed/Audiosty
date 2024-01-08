import parselmouth
from parselmouth.praat import call

# Load the audio file
audio_file_path = r'assests/temp_audio.wav'  # Replace with your audio file path
sound = parselmouth.Sound(audio_file_path)
# Get the total duration of the speech
total_duration = call(sound, "Get total duration")
# Convert to Intensity
intensity = call(sound, "To Intensity", 50, 0.025, 0.050)

# Detect silent intervals using intensity (adjust threshold as needed)
silence_threshold = -25  # Intensity threshold for silence
silent_intervals = call(intensity, "To TextGrid (silences)", silence_threshold, 0.1, 0.1, "Silence", "Sounding")

# Calculate the number of pauses and their durations
tier_number = 1
number_of_pauses = call(silent_intervals, "Get number of intervals", tier_number)
pause_durations = []
for i in range(1, number_of_pauses + 1):
    start_time = call(silent_intervals, "Get starting point", tier_number, i)
    end_time = call(silent_intervals, "Get end point", tier_number, i)
    pause_duration = end_time - start_time
    pause_durations.append(pause_duration)


# Placeholder for number of words and syllables
number_of_words = 100  # Placeholder value
number_of_syllables = 200  # Placeholder value

# Calculate Speech Rate (words per minute)
speech_rate = (number_of_words / (total_duration + 1e-7))

# Calculate Articulation Rate (syllables per minute)
articulation_rate = (number_of_syllables / (total_duration + 1e-7)) * 60

# Calculate Mean Length of Utterance (in syllables)
mean_length_syllables = number_of_syllables / (number_of_words + 1e-7)

# Calculate Mean Length of Utterance (in seconds)
mean_length_seconds = total_duration / (number_of_words + 1e-7)

# Calculate Number of Pauses per Minute (Total time)
number_of_pauses_total_time = len(pause_durations) / (total_duration * 60 + 1e-7)

# Calculate Number of Pauses per Minute (Speaking time)
speaking_time = total_duration - sum(pause_durations)
number_of_pauses_speaking_time = len(pause_durations) / (speaking_time * 60 + 1e-7)

# Calculate Mean Pause Duration
mean_pause_duration = sum(pause_durations) / (len(pause_durations) + 1e-7)

# Calculate Mean Syllable Duration
mean_syllable_duration = total_duration / (number_of_syllables + 1e-7)

# Calculate Phonation Time Ratio
phonation_time_ratio = speaking_time / (total_duration + 1e-7)

# Print the calculated values
print("Speech Rate:", total_duration)
print("Speech Rate:", speech_rate)
print("Articulation Rate:", articulation_rate)
print("Mean Length of Utterance (in syllables):", mean_length_syllables)
print("Mean Length of Utterance (in seconds):", mean_length_seconds)
print("Number of Pauses per Minute (Total time):", number_of_pauses_total_time)
print("Number of Pauses per Minute (Speaking time):", number_of_pauses_speaking_time)
print("Mean Pause Duration:", mean_pause_duration)
print("Mean Syllable Duration:", mean_syllable_duration)
print("Phonation Time Ratio:", phonation_time_ratio)
