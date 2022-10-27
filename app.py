import gradio as gr

img_to_text = gr.Blocks.load(name="spaces/pharma/CLIP-Interrogator")
text_to_music = gr.Interface.load("spaces/Mubert/Text-to-Music")

def get_prompts(uploaded_image):
  prompt = img_to_text(uploaded_image, fn_index=1)[0]
  music_result = get_music(prompt)
  return music_result

def get_music(prompt):
  email = "blabla@mail.com"
  duration = 30
  result = text_to_music(email, prompt, duration, loop=False)[0]
  return result

with gr.Blocks() as demo:
  with gr.Row():
    with gr.Column():
      input_img = gr.Image(type="filepath")
      generate = gr.Button("Generate Music from Image")
    with gr.Column():
      music_output = gr.Audio(label="Result")
  generate.click(get_prompts, inputs=[input_img], outputs=[music_output]

demo.launch()