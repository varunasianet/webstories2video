import io
import os
import hashlib
import threading
import markdown
import re
import json
import logging
from tortoise.utils.text import split_and_recombine_text
from fastapi import FastAPI, Response, Request, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from scipy.io.wavfile import write
import numpy as np
import ljinference
import msinference
import torch
import yaml
from fastapi.middleware.cors import CORSMiddleware
from decimal import Decimal
import phonemizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_using_gpu():
    return torch.cuda.is_available() and torch.cuda.current_device() >= 0

voice_path = "voices/"

# Load GPU config from file
try:
    with open('gpu_config.yml', 'r') as file:
        gpu_config = yaml.safe_load(file)
    gpu_device_id = gpu_config.get('gpu_device_id', 0)
except Exception as e:
    logger.error(f"Failed to load GPU config: {e}")
    gpu_device_id = 999  # Default to CPU if config can't be loaded

# Check if CUDA is available
if torch.cuda.is_available() and gpu_device_id != 999:
    torch.cuda.set_device(gpu_device_id)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

logger.info(f"Selected device: {device}")

def find_wav_files(directory):
    wav_files = []
    try:
        files = os.listdir(directory)
        for file in files:
            if file.lower().endswith(".wav"):
                file_name_without_extension = os.path.splitext(file)[0]
                wav_files.append(file_name_without_extension)
        wav_files.sort()
    except Exception as e:
        logger.error(f"Error finding WAV files: {e}")
    return wav_files

voicelist = find_wav_files(voice_path)
voices = {}

global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)

logger.info("Computing voices")
for v in voicelist:
    try:
        voices[v] = msinference.compute_style(f'voices/{v}.wav')
    except Exception as e:
        logger.error(f"Error computing style for voice {v}: {e}")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def index():
    try:
        with open('API_DOCS.md', 'r') as f:
            return Response(content=markdown.markdown(f.read()), media_type="text/html")
    except Exception as e:
        logger.error(f"Error reading API docs: {e}")
        raise HTTPException(status_code=500, detail="API documentation unavailable")

@app.get("/speakers")
async def get_speakers():
    speakers_special = []
    for speaker in voicelist:
        preview_url = f"/sample/{speaker}.wav"
        speaker_special = {
            'name': speaker,
            'voice_id': speaker,
            'preview_url': preview_url
        }
        speakers_special.append(speaker_special)
    return speakers_special

@app.get('/sample/{filename}')
async def get_sample(filename: str):
    file_path = os.path.join(voice_path, filename)
    if os.path.isfile(file_path):
        return FileResponse(file_path, media_type='audio/wav')
    else:
        logger.error(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")

@app.post("/api/v1/static")
async def serve_wav(
    text: str = Form(...),
    voice: str = Form(...),
    alpha: float = Form(0.3),
    beta: float = Form(0.7),
    diffusion_steps: int = Form(15),
    embedding_scale: float = Form(1.0)
):
    if voice.lower() not in voices:
        raise HTTPException(status_code=400, detail="Invalid voice selected")
    
    v = voices[voice.lower()]
    texts = split_and_recombine_text(text, 25, 225)
    audios = []
    
    try:
        for t in texts:
            audios.append(msinference.inference(t, v, alpha, beta, diffusion_steps, embedding_scale))
        
        output_buffer = io.BytesIO()
        write(output_buffer, 24000, np.concatenate(audios))
        output_buffer.seek(0)
        
        response = Response(content=output_buffer.getvalue(), media_type="audio/wav")
        response.headers["X-Using-GPU"] = str(is_using_gpu())
        return response
    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate audio")

