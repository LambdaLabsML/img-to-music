import gradio as gr
import os

img_to_text = gr.Blocks.load(name="spaces/pharma/CLIP-Interrogator")
text_to_music = gr.Interface.load("spaces/fffiloni/text-2-music")

def get_prompts(uploaded_image):
  
  print(f"""—————
  Calling CLIP Interrogator ...
  """)
  
  prompt = img_to_text(uploaded_image, fn_index=1)[0]
  
  music_result = get_music(prompt)
  
  return music_result

def get_music(prompt):
  
  print(f"""—————
  Calling now MubertAI ...
  ———————
  """)
  
  result = text_to_music(prompt, fn_index=0)
  
  print(f"""—————
  NEW RESULTS
  prompt : {prompt}
  music : {result}
  ———————
  """)
  
  return result, result

css = """
#col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
a {text-decoration-line: underline; font-weight: 600;}
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
      
  generate.click(get_prompts, inputs=[input_img], outputs=[music_output, output_text])

demo.queue(max_size=32, concurrency_count=20).launch()