from flask import Flask, render_template, request, jsonify
import torch
from diffusers import FluxPipeline
import os

app = Flask(__name__)

# Load the Flux model
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()  # Save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power.

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    prompt = request.form.get('prompt')
    
    # Generate image
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]

    # Save image
    image_path = os.path.join('static', 'generated_image.png')
    image.save(image_path)

    return jsonify({'image_path': image_path})

if __name__ == '__main__':
    app.run(debug=True)
