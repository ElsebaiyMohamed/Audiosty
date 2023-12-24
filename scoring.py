import numpy as np
import librosa

def calculate_expected_value(scores):
    # First calculate the probability of each unique score
    unique_scores, counts = np.unique(scores, return_counts=True)
    probabilities = counts / len(scores)
    
    # Then calculate the expected value as the sum of scores times their probabilities   
    expected_value = np.dot(unique_scores, probabilities)
    return expected_value


def calculate_fluency_score(audio_path, total_words, word_pronunciation_scores, base_script_len): 
    
    avg_pronunciation_score = calculate_expected_value(word_pronunciation_scores)
    if (total_words / base_script_len) < 0.15 or avg_pronunciation_score < 1.5:
        return 10
    audio, sr = librosa.load(audio_path)
    non_silent_intervals = librosa.effects.split(audio, top_db=22)
    non_silent_duration = sum([intv[1] - intv[0] for intv in non_silent_intervals]) / sr
    
    total_duration = len(audio) / sr
    
    non_silent_duration = non_silent_duration
    ideal_min_rate, ideal_max_rate = 120 / 60, 140 / 60
    actual_speech_rate = (total_words / (non_silent_duration + 1e-7)) * (total_words / base_script_len)
    speaking_ratio = non_silent_duration / total_duration
    # Existing speech rate score calculation
 
    # Determine if speech rate is within the ideal range
    if actual_speech_rate <= ideal_max_rate:
        # Within the ideal range or speaking slow
        max_ratio = actual_speech_rate / ideal_max_rate
        min_ratio = (actual_speech_rate / ideal_min_rate)
        speech_rate_score = np.mean([max_ratio, min_ratio]) - 0.167 
        # for normal speaking speech_rate_score between (0.708, 1) and for slow speaking speech_rate_score (0.707, 0)
    else:
        # Too fast 
        # for fast speaking speech_rate_score (0.707, 0)
        max_ratio = actual_speech_rate / ideal_max_rate
        speech_rate_score = 0.7 / max_ratio
    
    # If speaking ratio is significantly less than the gold standard, reduce the fluency score
    gold_standard_ratio = 0.9  # Assuming 90% speaking time is gold standard for natural speech
    speaking_ratio_score = min(speaking_ratio / gold_standard_ratio, 1)
    

    # Pronunciation score calculation
    avg_pronunciation_score = (avg_pronunciation_score - 1) / 2

    # pronunciation_variance = np.var(word_pronunciation_scores, ddof=1,)

    # Weighted combination of scores
    # Adjust weights as needed
    weight_speech_rate = 0.30
    weight_speaking_ratio = 0.20
    weight_pronunciation = 0.50
    # weight_pronunciation_variance = 0.10

    combined_score = speech_rate_score * weight_speech_rate + speaking_ratio_score * weight_speaking_ratio + avg_pronunciation_score * weight_pronunciation 
    
    # Scale the combined score to be between 10% and 100%
    scaled_fluency_score = 10 + combined_score * 80

    return scaled_fluency_score

def calculate_pronunciation_accuracy(word_pronunciation_scores, fluency_score, base_script_len, total_words):
    # if total_words / base_script_len < 0.25:
    #     return 10
    # Calculate average word pronunciation score
    avg_pronunciation_score = calculate_expected_value(word_pronunciation_scores)

    fluency_score = fluency_score / 100

    avg_pronunciation_score = (avg_pronunciation_score - 1) / 2
    avg_weight = 0.8
    flu_weight = 0.2
    combined_score = avg_weight * avg_pronunciation_score + flu_weight * fluency_score
    # Scale to 10% - 90%
    final_score = 10 + combined_score * 90  

    return final_score

def calculate_fluency_and_pronunciation(audio_path, total_words, word_pronunciation_scores, base_script_len):
    
    fluency_score = calculate_fluency_score(audio_path, total_words, word_pronunciation_scores, base_script_len)
    
    pronunciation_accuracy = calculate_pronunciation_accuracy(word_pronunciation_scores, fluency_score, base_script_len, total_words)
    
    return {'fluency_score': fluency_score, 'pronunciation_accuracy': pronunciation_accuracy}
    

if __name__ == '__main__':
    pass