from g2p_en import G2p
from string import punctuation 
from scoring import calculate_fluency_and_pronunciation

import whisper 
import torch 

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = whisper.load_model("base.en", device=device)

def transcribe(audio):
    result = model.transcribe(audio, word_timestamps=False, no_speech_threshold=0.4,  compression_ratio_threshold=2, temperature=0)
    return {'language': result['language'], 'text': result['text']}

def remove_punctuation(text):
    return ''.join(list(map(lambda x: '' if x in punctuation else x, text)))
def text2phoneme(text):
    g2p = G2p()
    text = text.lower()
    text = remove_punctuation(text)
    phonemes = g2p(text)
    phonemes.append(' ')
    new_phones = list()
    word = []
    for ph in phonemes:
        if ph == ' ':
            new_phones.append(word)
            word = []
        else:
            word.append(ph)
    return new_phones


def rate_pronunciation(expected_phonemes, actual_phonemes):
    expected_phonemes = expected_phonemes
    actual_phonemes = actual_phonemes
    # Calculate the Levenshtein distance between the two phoneme sequences
    results = []
    for i, base_word in enumerate(actual_phonemes): # ['ss', 'ad', ss]
        best_dist = float('inf')
        if i <= len(expected_phonemes): 
            word_best_score = float('inf')
            for window_idx in range(i, i + min(2, len(expected_phonemes) - i)):
                expected_word = expected_phonemes[window_idx]
                missed = 0
                for b_ph, ex_ph in zip(base_word, expected_word):
                    if b_ph != ex_ph:
                        missed += 1
                len_error = abs(len(base_word) - len(expected_word))
                missed = missed * 0.5
                missed += len_error
                missed = missed / len(base_word)
                if missed < word_best_score:
                    word_best_score = missed
                if -0.000001 < word_best_score < 0.000001:
                    word_best_score = 0
                    break

        if word_best_score == 0:
           results.append(3) 
        elif word_best_score <= 0.40:
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
