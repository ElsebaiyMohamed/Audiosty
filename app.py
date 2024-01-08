from flask import Flask, request, jsonify
from flask_cors import CORS
from wav2vec_aligen import speaker_pronunciation_assesment
from pydub import AudioSegment


app = Flask(__name__)  
CORS(app)

@app.route('/analyze_audio', methods=['POST', 'GET'])
def analyze_audio():
    if 'audio' not in request.files:
        return "audio is missing", 400
    if 'text' not in request.files:
        return "text is missing", 400
    
    audio = request.files['audio']
    text = request.files['text']
    if text.filename == '' or audio.filename == '':
        return "No selected file", 400
    
    if audio and text:
        # You can add file saving logic here
        text = text.read()
        temp_filename = 'temp_audio.wav'
        # wavfile.write(temp_filename, audio[0], audio[1])
        audio = AudioSegment.from_file(audio)
        audio.export(temp_filename, format="wav")

        response = speaker_pronunciation_assesment(temp_filename, text)
        response['word_levels'] = dict(response['word_levels'])
      
        return jsonify(response), 200

    return "Error processing request", 400

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
    
    