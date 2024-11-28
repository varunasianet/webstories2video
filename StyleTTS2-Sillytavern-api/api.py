import io
import os
import hashlib
import threading
import markdown
import re
import json
import logging
from tortoise.utils.text import split_and_recombine_text
from flask import Flask, Response, request, jsonify, send_file, url_for
from scipy.io.wavfile import write
import numpy as np
import ljinference
import msinference
import torch
import yaml
from flask_cors import CORS
from decimal import Decimal
import phonemizer
from scipy.io.wavfile import write
import io
import numpy as np
from scipy.io.wavfile import write
from flask import Flask, Response, request, jsonify
from pydub import AudioSegment


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

import re


def preprocess_tts_text(text):
    # Ensure the text ends with proper punctuation
    if not text.strip().endswith(('.', '!', '?')):
        text = text.strip() + '.'
    return text


app = Flask(__name__)
cors = CORS(app)

@app.route("/")
def index():
    try:
        with open('API_DOCS.md', 'r') as f:
            return markdown.markdown(f.read())
    except Exception as e:
        logger.error(f"Error reading API docs: {e}")
        return "API documentation unavailable", 500

@app.get("/speakers")
def get_speakers():
    speakers_special = []
    for speaker in voicelist:
        preview_url = url_for('get_sample', filename=f"{speaker}.wav", _external=True)
        speaker_special = {
            'name': speaker,
            'voice_id': speaker,
            'preview_url': preview_url
        }
        speakers_special.append(speaker_special)
    return jsonify(speakers_special)

@app.get('/sample/<filename>')
def get_sample(filename: str):
    file_path = os.path.join(voice_path, filename)
    if os.path.isfile(file_path):
        return send_file(file_path, mimetype='audio/wav', as_attachment=True)
    else:
        logger.error(f"File not found: {file_path}")
        return "File not found", 404


app = Flask(__name__)
@app.route("/api/v1/static", methods=['POST'])
def serve_wav():
    if 'text' not in request.form or 'voice' not in request.form:
        error_response = {'error': 'Missing required fields. Please include "text" and "voice" in your request.'}
        return jsonify(error_response), 400
    
    text = request.form['text'].strip()
    voice = request.form['voice'].strip().lower()
    alpha_float = float(request.form.get('alpha', '0.3'))
    beta_float = float(request.form.get('beta', '0.7'))
    diffusion_steps_int = int(request.form.get('diffusion_steps', '15'))
    embedding_scale_float = float(request.form.get('embedding_scale', '1'))
    speed_float = float(request.form.get('speed', '1.0'))

    if voice not in voices:
        error_response = {'error': 'Invalid voice selected'}
        return jsonify(error_response), 400
    
    v = voices[voice]
    
    # Split long text into chunks
    texts = split_and_recombine_text(text)
    audios = []
    
    try:
        for i, t in enumerate(texts):
            # Preprocess the text chunk
            processed_text = preprocess_tts_text(t)
            
            # For all chunks except the last one, add a short pause
            if i < len(texts) - 1:
                processed_text += ' '  # Add a space to create a slight pause
            
            audio = msinference.inference(processed_text, v, alpha_float, beta_float, diffusion_steps_int, embedding_scale_float, speed_float)
            audios.append(audio)

            # Clear intermediate tensors to free up GPU memory
            del audio  
            print("GPU SUMMARY",torch.cuda.memory_summary())
        
        # Concatenate all processed chunks
        final_audio = np.concatenate(audios)
        
        # Normalize audio
        final_audio = np.int16(final_audio / np.max(np.abs(final_audio)) * 32767)
        
        # Convert to AudioSegment
        audio_segment = AudioSegment(
            final_audio.tobytes(),
            frame_rate=24000,
            sample_width=2,
            channels=1
        )
        
        # Add silence padding at the end
        silence_duration_ms = 500  # 500ms of silence
        silence = AudioSegment.silent(duration=silence_duration_ms)
        audio_segment += silence
        
        # Export as WAV
        output_buffer = io.BytesIO()
        audio_segment.export(output_buffer, format="wav")
        output_buffer.seek(0)
        
        response = Response(output_buffer.getvalue())
        response.headers["Content-Type"] = "audio/wav"
        response.headers["X-Using-GPU"] = str(is_using_gpu())
        return response
    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        return jsonify({'error': 'Failed to generate audio'}), 500
