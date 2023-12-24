from phonemizer.separator import Separator
from phonemizer import phonemize
# from phonemizer.backend.espeak.wrapper import EspeakWrapper
from Levenshtein import distance as levenshtein_distance    
from scoring import calculate_fluency_and_pronunciation

import whisper 
import torch 

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = whisper.load_model("base.en", device=device)
separator = Separator(phone=None, word='',)

# EspeakWrapper.set_library(r"C:\Program Files\eSpeak NG\libespeak-ng.dll")

def transcribe(audio):
    result = model.transcribe(audio, word_timestamps=False, no_speech_threshold=0.4,  compression_ratio_threshold=2, temperature=0)
    return {'language': result['language'], 'text': result['text']}

def text2phoneme(text):
    return phonemize(text.lower().split(), backend='espeak' , separator=separator, strip=True, with_stress=False, tie=False, language='en-us')

def rate_pronunciation(expected_phonemes, actual_phonemes):
    expected_phonemes = expected_phonemes
    actual_phonemes = actual_phonemes
    # Calculate the Levenshtein distance between the two phoneme sequences
    results = []
    for i, base_word in enumerate(actual_phonemes):
        best_dist = float('inf')
        if i <= len(expected_phonemes): 
            for j in range(max(0, i-1), i + min(3, len(expected_phonemes) - i)):
                dist = levenshtein_distance(expected_phonemes[j], base_word,)
                if dist < best_dist:
                    best_dist = dist
                if best_dist == 0:  # Early stopping on perfect match
                    break
        error_threshold = len(base_word) * 0.40
        if best_dist == 0:
           results.append(3) 
        elif best_dist <= error_threshold:
            results.append(2) 
        else:
            results.append(1) 
    return results




def Speaker_speech_analysis(audio_path, text):
    pre_transcribtion = transcribe(audio_path)['text']
    print(pre_transcribtion)
    transcribtion = text2phoneme(pre_transcribtion)
    text_phone    = text2phoneme(text)
    scores        = rate_pronunciation(transcribtion, text_phone)
    FP_scores     = calculate_fluency_and_pronunciation(audio_path, len(pre_transcribtion.split()), scores, len(text.split()))
    word_scores = [(word, s) for word, s in zip(text.split(), scores)]
    
    FP_scores['word_scores'] = word_scores
    return FP_scores

if __name__ == '__main__':
    
    text = 'i have ADHD '
    text = text2phoneme(text)
    file_path = r'user_recording.wav'
    trans = transcribe(file_path)['text']
    print(trans)
    trans = text2phoneme(trans)
    print('base:', text)
    print('predicted:', trans)
    result = rate_pronunciation(trans, text)
    print(result)
