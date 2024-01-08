import librosa
import numpy as np
import os
import scipy.stats as stats


def normalize_aspect(value, mean, std):
    """ Normalize an aspect of speech using normal distribution. """
    return stats.norm(mean, std).cdf(value)

def calculate_fluency_scores(audio, total_words, content_score, pron_score):
    # Constants
    sample_rate = 16000  # Assuming a sample rate of 16 kHz
    # Define means and standard deviations for fluent speech
    speech_rate_mean, speech_rate_std = 150, 40
    phonation_time_mean, phonation_time_std = 50, 4

    # Calculate speaking and total duration
    non_silent_intervals = librosa.effects.split(audio, top_db=20)
    speaking_time = sum([intv[1] - intv[0] for intv in non_silent_intervals]) / sample_rate
    total_duration = len(audio) / sample_rate

    
    # Phonation time ratio
    phonation_time_ratio = speaking_time / total_duration * 60

    phonation_time_ratio = normalize_aspect(phonation_time_ratio, phonation_time_mean, phonation_time_std)
    if phonation_time_ratio > 0.5: 
        phonation_time_ratio =  0.5 - (phonation_time_ratio - 0.5)
    phonation_time_ratio = (phonation_time_ratio / 0.5) * 1
    
    
    speech_rate = (total_words / total_duration) * 60

    speech_rate_score = normalize_aspect(speech_rate, speech_rate_mean, speech_rate_std)
    if speech_rate_score > 0.5: 
        speech_rate_score =  0.5 - (speech_rate_score - 0.5)

    speech_rate_score = (speech_rate_score / 0.5) * 1
    
    content_score, pron_score = content_score / 100, pron_score / 100

    w_rate_score = 0.45
    w_pho_ratio  = 0.30
    w_cont         = 0.05
    w_pro           = 0.2
    scaled_fluency_score = speech_rate_score * w_rate_score + phonation_time_ratio * w_pho_ratio + content_score * w_cont + pron_score * w_pro
    return scaled_fluency_score


a, _ = librosa.load(r"D:\test cases\Test 1\33\33-RA-Reptile.wav", sr=16000)
text = 'Around 250 million years ago, 700 species of reptiles closely related to the modern-day crocodile roamed the earth, now new research reveals how a complex interplay between climate change, species competition and habitat can help explain why just 23 species of crocodile survive today.'

word_count = len(text.split())
content = 95

print(calculate_fluency_scores(a, word_count, content))