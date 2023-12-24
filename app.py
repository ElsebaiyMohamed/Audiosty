import gradio as gr
from logic import Speaker_speech_analysis
from scipy.io import wavfile



def create_html_from_scores(word_scores):
    html_output = ''
    for word, score in word_scores:
        if score == 1:
            html_output += f'<span style="color: #dc3545;">{word}</span> '
        elif score == 2:
            html_output += f'<span style="color: #ffc107;">{word}</span> '
        else:
            html_output += f'<span style="color: #28a745;">{word}</span> '
    return html_output
  
def generate_progress_bar(score, label):
    score = round(score, 2)
    score_text = f"{score:.2f}" if score < 90 else "90"
    bar_color = "#dc3545" if score < 30 else "#ffc107" if score < 60 else "#28a745"
    bar_length = f"{(score / 90) * 100}%"
    return f"""
    <div class="progress-label">{label}:</div>
    <div class="progress-container">
        <div class="progress-bar" style="width: {bar_length}; background-color: {bar_color};">
            <div class="progress-score">{score_text}</div>
        </div>
    </div>
    <div class="progress-max">Max: 90</div>
    """
# CSS to be used in the Gradio Interface




def analyze_audio(text, audio):
# Write the processed audio to a temporary WAV file
    temp_filename = 'temp_audio.wav'
    wavfile.write(temp_filename, audio[0], audio[1])


    result = Speaker_speech_analysis(temp_filename, text)
    accuracy_score = result['pronunciation_accuracy']
    fluency_score  = result['fluency_score']
    word_scores    = result['word_scores']
    
    html_content = create_html_from_scores(word_scores)
    pronunciation_progress_bar = generate_progress_bar(accuracy_score, "Pronunciation Accuracy")
    fluency_progress_bar = generate_progress_bar(fluency_score, "Fluency Score")
    
    
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
    </div>
    
    <p class="text">
      {html_content}
    </p>

    {pronunciation_progress_bar}
    {fluency_progress_bar}
    """
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
    iface.launch()