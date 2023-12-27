from flask import Flask, request, jsonify
from flask_cors import CORS
from logic import Speaker_speech_analysis
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

        result = Speaker_speech_analysis(temp_filename, text)
        accuracy_score = result['pronunciation_accuracy']
        fluency_score  = result['fluency_score']
        word_scores    = result['word_scores']
        response = {
                "Accuracy": accuracy_score,
                "Fluency": fluency_score,
                'Words_score': word_scores
            }
        return jsonify(response), 200

    return "Error processing request", 400

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
    
    