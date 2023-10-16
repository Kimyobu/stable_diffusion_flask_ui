from flask import Flask, render_template
from flask_socketio import SocketIO
import argparse
import diffusers
import torch
from preview_decoder import ApproximateDecoder
from io import BytesIO
import base64
import threading
import json

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, help="Port to be bind server.")
parser.add_argument("--model", type=str)

args = parser.parse_args()
PORT = args.port
MODEL = args.model

app = Flask("Stable diffusion")
socketio = SocketIO(app)

if torch.cuda.is_available() is False:
    raise EnvironmentError("Cuda is Not Available")

pipe = diffusers.StableDiffusionPipeline.from_pretrained(MODEL).to("cuda")
ApproxDec = ApproximateDecoder.for_pipeline(pipeline=pipe)

def image_to_data_uri(image):
    image_data = BytesIO()
    image.save(image_data, format='PNG')
    image_data.seek(0)
    
    base64_image = base64.b64encode(image_data.read()).decode('utf-8')
    data_uri = 'data:image/jpeg;base64,' + base64_image
    
    return data_uri

def display_latents_callback(step: int, timestep: int, latents: torch.FloatTensor):
    latents_image = ApproxDec(latents.squeeze(0))
    # แสดงรูปภาพพร้อม title เป็น step
    data_uri = image_to_data_uri(latents_image)
    socketio.emit('pre', data_uri, broadcast=True)
    

@socketio.on("gen")
def gen(data):
    print(f"Generate with: {json.dumps(data, indent=2)}")
    p = pipe(**data, callback=display_latents_callback)
    image = p.images[0]
    data_uri = image_to_data_uri(image)
    socketio.emit("complete_gen", image, broadcast=True)

@socketio.on("connect")
def connect():
    print("Client Has Connected!!")

@socketio.on("disconnect")
def disconnect():
    print("Client Has DisConnected :< Bye")

@app.route('/')
def index():
    with open("./index.html", encoding="utf-8") as f:
        page = f.read()
    return page


socketio.run(app, debug=True, allow_unsafe_werkzeug=True, port=PORT)

del pipe
torch.cuda.empty_cache()