import gradio as gr
import os
import requests
import urllib

from os import path
from pydub import AudioSegment

img_to_text = gr.Blocks.load(name="spaces/pharma/CLIP-Interrogator")
text_to_music = gr.Interface.load("spaces/fffiloni/text-2-music")

from share_btn import community_icon_html, loading_icon_html, share_js

def get_prompts(uploaded_image):
  
  prompt = img_to_text(uploaded_image, fn_index=1)[0]
  
  music_result = get_music(prompt)
  
  return music_result

def get_music(prompt):
  
  result = text_to_music(prompt, fn_index=0)
  
  print(f"""—————
  NEW RESULTS
  prompt : {prompt}
  music : {result}
  ———————
  """)
  
  url = result
  save_as = "file.mp3"
  
  data = urllib.request.urlopen(url)

  f = open(save_as,'wb')
  f.write(data.read())
  f.close()
  
  wave_file="file.wav"
  
  sound = AudioSegment.from_mp3(save_as)
  sound.export(wave_file, format="wav")
  
  return wave_file, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

css = """
#col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
a {text-decoration-line: underline; font-weight: 600;}
.animate-spin {
    animation: spin 1s linear infinite;
}
@keyframes spin {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
}
#share-btn-container {
    display: flex; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
}
#share-btn {
    all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;right:0;
}
#share-btn * {
    all: unset;
}
#share-btn-container div:nth-child(-n+2){
    width: auto !important;
    min-height: 0px !important;
}
#share-btn-container .wrap {
    display: none !important;
}
"""

with gr.Blocks(css=css) as demo:
  with gr.Column(elem_id="col-container"):
    gr.HTML("""<div style="text-align: center; max-width: 700px; margin: 0 auto;">
              <div
              style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
              "
              >
              <h1 style="font-weight: 900; margin-bottom: 7px; margin-top: 5px;">
                  Image to Music
              </h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 94%">
              Sends an image in to <a href="https://huggingface.co/spaces/pharma/CLIP-Interrogator" target="_blank">CLIP Interrogator</a>
              to generate a text prompt which is then run through 
              <a href="https://huggingface.co/Mubert" target="_blank">Mubert</a> text-to-music to generate music from the input image!
              </p>
          </div>""")
    
    
    input_img = gr.Image(type="filepath", elem_id="input-img")
    generate = gr.Button("Generate Music from Image")
  
    music_output = gr.Audio(label="Result", type="filepath", elem_id="music-output")
    
    with gr.Group(elem_id="share-btn-container"):
      community_icon = gr.HTML(community_icon_html, visible=False)
      loading_icon = gr.HTML(loading_icon_html, visible=False)
      share_button = gr.Button("Share to community", elem_id="share-btn", visible=False)
      
  generate.click(get_prompts, inputs=[input_img], outputs=[music_output, share_button, community_icon, loading_icon])
  share_button.click(None, [], [], _js=share_js)

demo.queue(max_size=32, concurrency_count=20).launch()