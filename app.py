import gradio as gr
# from logic import Speaker_speech_analysis
from scipy.io import wavfile
from wav2vec_aligen import speaker_pronunciation_assesment



def create_html_from_scores(word_levels):
    html_output = ''
    for word, level in word_levels:
        if level == '/':
            html_output += f'<span style="color: #0000ff;">{level}</span> '
        elif level == 'Wrong':
          html_output += f'<span style="color: #dc3545;">{word}</span> '
        elif level == 'Understandable':
          html_output += f'<span style="color: #ffc107;">{word}</span> '
        else:
            html_output += f'<span style="color: #28a745;">{word}</span> '
    return html_output
  
def generate_progress_bar(score, label):
    score = round(score, 2)
    score_text = f"{score:.2f}" if score < 100 else "100"
    if score < 30:
        bar_color = "#dc3545" 
    elif score < 60:
        bar_color = "#dc6545" 
    elif score < 80:
        bar_color = "#ffc107"
    else:
        bar_color = "#28a745"
    bar_length = f"{(score / 100) * 100}%"
    return f"""
    <div class="progress-label">{label}:</div>
    <div class="progress-container">
        <div class="progress-bar" style="width: {bar_length}; background-color: {bar_color};">
            <div class="progress-score">{score_text}</div>
        </div>
    </div>
    <div class="progress-max">Max: 100</div>
    """
# CSS to be used in the Gradio Interface




def analyze_audio(text, audio):
# Write the processed audio to a temporary WAV file
    if text is None or audio is None:
      return 'the audio or the text is missing'
    temp_filename = 'temp_audio.wav'
    wavfile.write(temp_filename, audio[0], audio[1])


    result = speaker_pronunciation_assesment(temp_filename, text)
    accuracy_score = result['pronunciation_accuracy']
    fluency_score   = result['fluency_score']
    word_levels      = result['word_levels']
    content_scores = result['content_scores']
    wpm                = result['wpm']
    
    html_content = create_html_from_scores(word_levels)
    pronunciation_progress_bar = generate_progress_bar(accuracy_score, "Pronunciation Accuracy")
    fluency_progress_bar = generate_progress_bar(fluency_score, "Fluency Score")
    content_progress_bar = generate_progress_bar(content_scores, "Content Score")
    
    
    html_with_css = f"""
    <style>
    .legend {{
      font-size: 22px;
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    
    .legend-dot {{
        height: 15px;
        width: 15px;
        border-radius: 50%;
        display: inline-block;
      }}
      
    .good {{ color: #28a745; 
    }}
    .average {{ color: #ffc107; 
    }}
    .bad {{ color: #dc3545;
    }}
    
    .wrong {{ color: #dc3545;
    }}
        
    .text {{
        font-size: 20px;
        margin-bottom: 20px;
      }}

    .progress-container {{
        width: 100%;
        background-color: #ddd;
        border-radius: 13px;
        overflow: hidden;
      }}

    .progress-bar {{
        height: 30px;
        line-height: 30px;
        text-align: center;
        font-size: 16px;
        border-radius: 15px;
        transition: width 1s ease;
      }}

    .progress-label {{
        font-weight: bold;
        font-size: 22px;
        margin-bottom: 20px;
        margin-top: 5px;
        text-align: center;
      }}

    .progress-score {{
        display: inline-block;
        color: black;
      }}

    .progress-max {{
        text-align: right;
        margin: 10px;
        font-size: 16px;
      }}
        
    </style>
    
    
    <div class="legend">
      <span class="legend-dot" style="background-color: #28a745;"></span><span>Good</span>
      <span class="legend-dot" style="background-color: #ffc107;"></span><span>Understandable</span>
      <span class="legend-dot" style="background-color: #dc3545;"></span><span>Bad</span>
      <span class="legend-dot" style="background-color: #0000ff;"></span><span>No Speech</span>
    </div>
    
    <p class="text">
      {html_content}
    </p>
    
    <p class="text">
      <span style="color: #0000ff;">Word Per Minute {wpm:0.2f}</span>
    </p>

    {pronunciation_progress_bar}
    {fluency_progress_bar}
    {content_progress_bar}
    """
        # 

    return html_with_css

# Define the Gradio interface
iface = gr.Interface(fn=analyze_audio,
                     inputs=[gr.Textbox(label='Training Text', placeholder='Write the text for pronunciation task', interactive=True, visible=True, show_copy_button=True,), 
                             gr.Audio(label="Recoreded Audio", sources=['microphone', 'upload'])
                             ],
                     outputs=[gr.HTML(label="Analysis of pronunciation"),
                              ],
                    #  css=additional_css,
                     # title="Audio Analysis Tool",
                     description="Write any text and recored an audio to predict pronunciation erors"
                     )

# Run the Gradio app
if __name__ == "__main__":
    iface.launch(share=True)