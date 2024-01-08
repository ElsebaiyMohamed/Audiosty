from dataclasses import dataclass
import torch
import librosa
import numpy as np
import os
import scipy.stats as stats

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
# os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = "C:\Program Files\eSpeak NG\libespeak-ng.dll"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from optimum.bettertransformer import BetterTransformer
torch.random.manual_seed(0);

model_name = "facebook/wav2vec2-lv-60-espeak-cv-ft"
processor = Wav2Vec2Processor.from_pretrained(model_name, phone_delimiter_token=' ', word_delimiter_token=' ')
model = Wav2Vec2ForCTC.from_pretrained(model_name).to('cpu').eval()
model = BetterTransformer.transform(model)

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float
    
# Merge the labels
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d}]"

    def __len__(self):
        return self.end - self.start

def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    trellis = torch.zeros((num_frame, num_tokens))
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1 :, 0] = float("inf")

    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(trellis[t, 1:] + emission[t, blank_id], # Score for staying at the same token
                                           trellis[t, :-1] + emission[t, tokens[1:]], # Score for changing to the next token
                                           )
    return trellis


def backtrack(trellis, emission, tokens, blank_id=0):
    t, j = trellis.size(0) - 1, trellis.size(1) - 1

    aligenment_path = [Point(j, t, emission[t, blank_id].exp().item())]
    while j > 0:
        # Should not happen but just in case
        assert t > 0

        # 1. Figure out if the current position was stay or change
        # Frame-wise score of stay vs change
        p_stay = emission[t - 1, blank_id]
        p_change = emission[t - 1, tokens[j]]

        # Context-aware score for stay vs change
        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change

        # Update position
        t -= 1
        if changed > stayed:
            j -= 1

        # Store the aligenment_path with frame-wise probability.
        prob = (p_change if changed > stayed else p_stay).exp().item()
        aligenment_path.append(Point(j, t, prob))

    # Now j == 0, which means, it reached the SoS.
    # Fill up the rest for the sake of visualization
    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        aligenment_path.append(Point(j, t - 1, prob))
        t -= 1

    return aligenment_path[::-1]

def merge_repeats(aligenment_path, ph):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(aligenment_path):
        while i2 < len(aligenment_path) and aligenment_path[i1].token_index == aligenment_path[i2].token_index:
            i2 += 1
        score = sum(aligenment_path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                ph[aligenment_path[i1].token_index],
                aligenment_path[i1].time_index,
                aligenment_path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments



def load_model(device='cpu'):
    model_name = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    processor = Wav2Vec2Processor.from_pretrained(model_name, phone_delimiter_token=' ', word_delimiter_token=' ')
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device).eval()
    model = BetterTransformer.transform(model)
    return processor, model

def load_audio(audio_path, processor):
    audio, sr = librosa.load(audio_path, sr=16000)

    input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values
    return input_values
        

@torch.inference_mode()
def get_emissions(input_values, model):
    emissions = model(input_values,).logits
    emissions = torch.log_softmax(emissions, dim=-1)
    emission = emissions[0].cpu().detach()
    return emission

def get_chnocial_phonemes(transcript, processor):
    transcript = transcript.replace('from the', 'from | the')
    phoneme_ids = processor.tokenizer(transcript).input_ids
    ph = processor.tokenizer.phonemize(transcript)
    phoneme_list = ph.replace('   ', ' ').split()
    transcript = transcript.replace('from | the', 'from the')
    words = transcript.split()
    words_phonemes = ph.split('   ')
    words_phoneme_mapping = [(w, p) for w, p in zip(words, words_phonemes)]


    return phoneme_list, phoneme_ids, words_phoneme_mapping


def word_level_scoring(words_phoneme_mapping, segments):
    word_scores = []
    start = 0
    for word, ph_seq in words_phoneme_mapping:
        n_ph = len(ph_seq.split())
        cum_score = 0
        wrong = 0
        for i in range(start, start + n_ph):
            s = segments[i]
            cum_score += s.score
            if s.score < 0.50:
                wrong += 1

        start += n_ph
        word_scores.append((word, np.round(cum_score / n_ph, 5), np.round(wrong / n_ph, 5)))
    return word_scores

def map_word2_class(word_scores):
    word_levels = []
    for w, sc, wrong_ratio in word_scores:
        if wrong_ratio > 0.5 or sc < 0.60:
            word_levels.append((w, '/'))
        elif sc < 0.70:
            word_levels.append((w, 'Wrong'))
        elif sc < 0.85:
            word_levels.append((w, 'Understandable'))
        else:
            word_levels.append((w, 'Correct'))
    return word_levels

def calculate_content_scores(word_levels):
    content_scores = len(word_levels)
    for w, c in word_levels:
        if c == '/':
            content_scores -= 1
        elif c == 'Wrong':
            content_scores -= 0.5
        else:None
    content_scores = (content_scores / len(word_levels)) * 100
    return content_scores
        
def calculate_sentence_pronunciation_accuracy(word_scores):
    w_scores = 0
    error_scores = 0
    for w, sc, wrong_ratio in word_scores:
        sc = sc * 100
        if sc > 60:
            if sc < 70:
                sc = ((sc - 60) / (70 - 60)) * (20 - 0)  + 0
            elif sc < 88:
                sc = ((sc - 70) / (88 - 70)) * (70 - 20)  + 20
            else:
                sc = ((sc - 88) / (100 - 88)) * (100 - 70)  + 70
        w_scores += sc
        error_scores += wrong_ratio
    w_scores = (w_scores / len(word_scores))
    # w_scores =( (w_scores - 50) / (100 - 50)) * 100 
    error_scores = (error_scores / len(word_scores)) * 40
    pronunciation_accuracy = min(w_scores, w_scores - error_scores)
    return pronunciation_accuracy

def get_hard_aligenment_with_scores(input_values, transcript):
    # processor, model = load_model(device='cpu')
    
    emission = get_emissions(input_values, model)
    phoneme_list, phoneme_ids, words_phoneme_mapping = get_chnocial_phonemes(transcript, processor)
    trellis = get_trellis(emission, phoneme_ids)
    aligenment_path = backtrack(trellis, emission, phoneme_ids)
    segments = merge_repeats(aligenment_path, phoneme_list)
    return segments, words_phoneme_mapping

def normalize_aspect(value, mean, std):
    """ Normalize an aspect of speech using normal distribution. """
    return stats.norm(mean, std).cdf(value)

def calculate_fluency_scores(audio, total_words, content_score, pron_score):
    # Constants
    content_score, pron_score = content_score / 100, pron_score / 100
    sample_rate = 16000  # Assuming a sample rate of 16 kHz
    # Define means and standard deviations for fluent speech
    speech_rate_mean, speech_rate_std = 170, 50
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
    
    
    speech_rate = (total_words / (total_duration / 60)) 
    speech_rate = speech_rate * content_score
    speech_rate_score = normalize_aspect(speech_rate, speech_rate_mean, speech_rate_std)
    if speech_rate_score > 0.5: 
        speech_rate_score =  0.5 - (speech_rate_score - 0.5)

    speech_rate_score = (speech_rate_score / 0.5) * 1
    

    w_rate_score = 0.4
    w_pho_ratio  = 0.35
    w_pro           = 0.25
    scaled_fluency_score = speech_rate_score * w_rate_score + phonation_time_ratio * w_pho_ratio + pron_score * w_pro
    scaled_fluency_score = scaled_fluency_score * 100
    return scaled_fluency_score, speech_rate



def speaker_pronunciation_assesment(audio_path, transcript):
    input_values = load_audio(audio_path, processor)
    segments, words_phoneme_mapping = get_hard_aligenment_with_scores(input_values, transcript)
    word_scores = word_level_scoring(words_phoneme_mapping, segments)
    word_levels = map_word2_class(word_scores)
    content_scores = calculate_content_scores(word_levels)
    pronunciation_accuracy = calculate_sentence_pronunciation_accuracy(word_scores)
    fluency_accuracy, wpm = calculate_fluency_scores(input_values[0], len(word_scores), content_scores, pronunciation_accuracy) 
    

    result = {'pronunciation_accuracy': pronunciation_accuracy,
              'word_levels': word_levels, 
              'content_scores': content_scores,
              'wpm': wpm,
              'stress': None,
              'fluency_score': fluency_accuracy}
    return result

if __name__ == '__main__':
    MODEL_IS_LOADED = False
else:
    MODEL_IS_LOADED = False

