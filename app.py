import gradio as gr
import os

#img_to_text = gr.Blocks.load(name="spaces/pharma/CLIP-Interrogator")
text_to_music = gr.Interface.load("spaces/fffiloni/text-2-music")

def get_prompts(uploaded_image):
  print("calling Clip interrogator ...")
  
  #prompt = img_to_text(uploaded_image, fn_index=1)[0]
  prompt = "trying to find the right method index"
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
  #output = os.path.join(result)
  #print(output)
  return result

with gr.Blocks() as demo:
  with gr.Row():
    with gr.Column():
      input_img = gr.Image(type="filepath")
      generate = gr.Button("Generate Music from Image")
    with gr.Column():
      music_output = gr.Textbox(label="Result")
  generate.click(get_prompts, inputs=[input_img], outputs=[music_output])

demo.launch()