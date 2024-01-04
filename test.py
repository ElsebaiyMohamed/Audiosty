from allosaurus.app import read_recognizer

# load your model by the <model name>, will use 'latest' if left empty
model = read_recognizer('latest')

# # run inference on <audio_file> with <lang>, lang will be 'ipa' if left empty
# y = model.recognize(r"C:\Users\20101\Downloads\LDC93S1.wav", 'ipa', timestamp=False)
# print(y)
# print()

# y = model.recognize(r"C:\Users\20101\Downloads\LDC93S1.wav", 'eng' , timestamp=False)
# print(y)

import phonemizer
text = 'Hello'
ipa = phonemizer.phonemize(text,)

print(ipa)