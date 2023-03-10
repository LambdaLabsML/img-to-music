import time
import base64
import gradio as gr
from sentence_transformers import SentenceTransformer

import httpx
import json

import os
import requests
import urllib

from os import path
from pydub import AudioSegment

#img_to_text = gr.Blocks.load(name="spaces/pharma/CLIP-Interrogator")
img_to_text = gr.Blocks.load(name="spaces/fffiloni/CLIP-Interrogator-2")

from share_btn import community_icon_html, loading_icon_html, share_js

def get_prompts(uploaded_image, track_duration, gen_intensity, gen_mode):
  print("calling clip interrogator")
  #prompt = img_to_text(uploaded_image, "ViT-L (best for Stable Diffusion 1.*)", "fast", fn_index=1)[0]
  prompt = img_to_text(uploaded_image, 'fast', 4, fn_index=1)[0]
  print(prompt)
  music_result = generate_track_by_prompt(prompt, track_duration, gen_intensity, gen_mode)
  print(music_result)
  return music_result[0], gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

from utils import get_tags_for_prompts, get_mubert_tags_embeddings, get_pat

minilm = SentenceTransformer('all-MiniLM-L6-v2')
mubert_tags_embeddings = get_mubert_tags_embeddings(minilm)


def get_track_by_tags(tags, pat, duration, gen_intensity, gen_mode, maxit=20):
    
    r = httpx.post('https://api-b2b.mubert.com/v2/RecordTrackTTM',
                   json={
                       "method": "RecordTrackTTM",
                       "params": {
                           "pat": pat,
                           "duration": duration,
                           "format": "wav",
                           "intensity":gen_intensity,
                           "tags": tags,
                           "mode": gen_mode
                       }
                   })

    rdata = json.loads(r.text)
    assert rdata['status'] == 1, rdata['error']['text']
    trackurl = rdata['data']['tasks'][0]['download_link']

    print('Generating track ', end='')
    for i in range(maxit):
        r = httpx.get(trackurl)
        if r.status_code == 200:
            return trackurl
        time.sleep(1)


def generate_track_by_prompt(prompt, duration, gen_intensity, gen_mode):
    try:
        pat = get_pat("prodia@prodia.com")
        _, tags = get_tags_for_prompts(minilm, mubert_tags_embeddings, [prompt, ])[0]
        result = get_track_by_tags(tags, pat, int(duration), gen_intensity, gen_mode)
        print(result)
        return result, ",".join(tags), "Success"
    except Exception as e:
        return None, "", str(e)

def convert_mp3_to_wav(mp3_filepath):
 
  url = mp3_filepath
  save_as = "file.mp3"
  
  data = urllib.request.urlopen(url)

  f = open(save_as,'wb')
  f.write(data.read())
  f.close()
  
  wave_file="file.wav"
  
  sound = AudioSegment.from_mp3(save_as)
  sound.export(wave_file, format="wav")
  
  return wave_file

article = """
    
    <div class="footer">
        <p>
         
        Follow <a href="https://twitter.com/fffiloni" target="_blank">Sylvain Filoni</a> for future updates ????
        </p>
    </div>
    
    <div id="may-like-container" style="display: flex;justify-content: center;flex-direction: column;align-items: center;margin-bottom: 30px;">
        <p style="font-size: 0.8em;margin-bottom: 4px;">You may also like: </p>
        <div id="may-like" style="display: flex;flex-wrap: wrap;align-items: center;height: 20px;">
            <svg height="20" width="122" style="margin-left:4px;margin-bottom: 6px;">       
                 <a href="https://huggingface.co/spaces/fffiloni/spectrogram-to-music" target="_blank">
                    <image href="https://img.shields.io/badge/???? Spaces-Riffusion-blue" src="https://img.shields.io/badge/???? Spaces-Riffusion-blue.png" height="20"/>
                 </a>
            </svg>
            <svg height="20" width="192" style="margin-left:4px;margin-bottom: 6px;">       
                 <a href="https://huggingface.co/spaces/Mubert/Text-to-Music" target="_blank">
                    <image href="https://img.shields.io/badge/???? Spaces-Mubert_Text_to_Music-blue" src="https://img.shields.io/badge/???? Spaces-Mubert_Text_to_Music-blue.png" height="20"/>
                 </a>
            </svg>
        </div>
    </div>

    
"""

with gr.Blocks(css="style.css") as demo:
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
        music_output = gr.Audio(label="Result", type="filepath", elem_id="music-output").style(height="5rem")
        
        with gr.Group(elem_id="share-btn-container"):
            community_icon = gr.HTML(community_icon_html, visible=False)
            loading_icon = gr.HTML(loading_icon_html, visible=False)
            share_button = gr.Button("Share to community", elem_id="share-btn", visible=False)

        with gr.Accordion(label="Music Generation Options", open=False):
            track_duration = gr.Slider(minimum=20, maximum=120, value=30, step=5, label="Track duration", elem_id="duration-inp")
            with gr.Row():
                gen_intensity = gr.Dropdown(choices=["low", "medium", "high"], value="medium", label="Intensity")
                gen_mode = gr.Radio(label="mode", choices=["track", "loop"], value="track")
        
        generate = gr.Button("Generate Music from Image")

        gr.HTML(article)
    
    generate.click(get_prompts, inputs=[input_img,track_duration,gen_intensity,gen_mode], outputs=[music_output, share_button, community_icon, loading_icon], api_name="i2m")
    share_button.click(None, [], [], _js=share_js)

demo.queue(max_size=32, concurrency_count=20).launch()