import gradio as gr
import os

img_to_text = gr.Blocks.load(name="spaces/pharma/CLIP-Interrogator")
text_to_music = gr.Interface.load("spaces/fffiloni/text-2-music")

def get_prompts(uploaded_image):
  print("calling Clip interrogator ...")
  
  prompt = img_to_text(uploaded_image, fn_index=1)[0]
 
  print(f"""———
  Got prompt result:
  {prompt}
  ———————
  """)
  
  music_result = get_music(prompt)
  
  return music_result

def get_music(prompt):
  print("calling now mubert ....")
  result = text_to_music(prompt, fn_index=0)
  print(result)
  return result

with gr.Blocks() as demo:
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
            Mubert text-to-music to generate music from the input image!
            </p>
        </div>""")
  with gr.Row():
    with gr.Column():
      input_img = gr.Image(type="filepath")
      generate = gr.Button("Generate Music from Image")
    with gr.Column():
      music_output = gr.Audio(label="Result", type="filepath")
  generate.click(get_prompts, inputs=[input_img], outputs=[music_output])

demo.queue(max_size=32).launch()