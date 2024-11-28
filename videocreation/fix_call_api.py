import requests
import json
import sqlalchemy
from google.cloud.sql.connector import Connector
import logging
from datetime import datetime
import pytz
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define IST timezone
ist = pytz.timezone('Asia/Kolkata')

# API endpoint
VIDEO_POST_API_URL = "https://core-webservices-api.asianetnews.com/ai/video/add"

# Database connection details
INSTANCE_CONNECTION_NAME = "asianet-tech-staging:asia-south1:webstories-asianet-db"
DB_USER = "webstories-user"
DB_PASS = "asianetweb"
DB_NAME = "webstoriesrss"

# Initialize database connection
def init_connection_engine():
    logger.info("Initializing database connection...")
    connector = Connector()

    def getconn():
        conn = connector.connect(
            INSTANCE_CONNECTION_NAME,
            "pymysql",
            user=DB_USER,
            password=DB_PASS,
            db=DB_NAME,
        )
        return conn

    engine = sqlalchemy.create_engine(
        "mysql+pymysql://",
        creator=getconn,
    )
    logger.info("Database connection initialized successfully.")
    return engine

# Function to get webstories with non-empty video paths
def get_webstories_with_videos(engine):
    logger.info("Fetching webstories with non-empty video paths...")
    with engine.connect() as connection:
        query = sqlalchemy.text("""
            SELECT webstory_id, link, vertical_video_path, horizontal_video_path
            FROM webstories
            WHERE vertical_video_path IS NOT NULL 
            AND vertical_video_path != ''
            AND horizontal_video_path IS NOT NULL 
            AND horizontal_video_path != ''
        """)
        result = connection.execute(query)
        webstories = result.fetchall()
    logger.info(f"Found {len(webstories)} webstories with non-empty video paths.")
    return webstories

# Function to update posted_to_api status
def update_posted_to_api_status(engine, webstory_id, status):
    logger.info(f"Updating posted_to_api status for webstory ID {webstory_id} to {status}")
    with engine.connect() as connection:
        query = sqlalchemy.text("""
            UPDATE webstories
            SET posted_to_api = :status
            WHERE webstory_id = :webstory_id
        """)
        connection.execute(query, {
            "status": status,
            "webstory_id": webstory_id
        })
        connection.commit()
    logger.info(f"Status updated successfully for webstory ID {webstory_id}")

# Function to post video URLs to API
def post_video_urls(engine, webstory_id, article_url, vertical_video_url, horizontal_video_url):
    logger.info(f"Posting video URLs for webstory ID {webstory_id}")
    logger.info(f"Article URL: {article_url}")
    logger.info(f"Vertical Video URL: {vertical_video_url}")
    logger.info(f"Horizontal Video URL: {horizontal_video_url}")

    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "articleUrl": article_url,
        "vertical_video": vertical_video_url,
        "horizontal_video": horizontal_video_url
    }
    try:
        logger.info("Sending POST request to API...")
        response = requests.post(VIDEO_POST_API_URL, json=data, headers=headers, timeout=30)
        logger.info(f"API Response Status Code: {response.status_code}")
        logger.info(f"API Response Content: {response.text}")
        
        response.raise_for_status()
        logger.info(f"Successfully posted video URLs for webstory ID: {webstory_id}")
        update_posted_to_api_status(engine, webstory_id, 'YES')
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred while posting video URLs for webstory ID {webstory_id}: {str(e)}")
        if hasattr(e, 'response'):
            logger.error(f"Error Response Status Code: {e.response.status_code}")
            logger.error(f"Error Response Content: {e.response.text}")
        update_posted_to_api_status(engine, webstory_id, 'NO')
        return False

# Main function to process and post webstories
def process_and_post_webstories():
    logger.info("Starting to process and post webstories...")
    engine = init_connection_engine()
    webstories = get_webstories_with_videos(engine)
    
    total_webstories = len(webstories)
    successful_posts = 0
    failed_posts = 0

    for index, webstory in enumerate(webstories, 1):
        webstory_id, article_url, vertical_path, horizontal_path = webstory
        
        logger.info(f"Processing webstory {index} of {total_webstories}")
        logger.info(f"Webstory ID: {webstory_id}")

        # Use the paths directly without adding any prefix
        vertical_video_url = vertical_path
        horizontal_video_url = horizontal_path
        
        success = post_video_urls(engine, webstory_id, article_url, vertical_video_url, horizontal_video_url)
        
        if success:
            logger.info(f"Successfully processed and posted webstory ID: {webstory_id}")
            successful_posts += 1
        else:
            logger.warning(f"Failed to post webstory ID: {webstory_id}")
            failed_posts += 1
        
        logger.info(f"Waiting for 10 seconds before processing the next webstory...")
        time.sleep(10)

    logger.info("Finished processing all webstories.")
    logger.info(f"Total webstories processed: {total_webstories}")
    logger.info(f"Successful posts: {successful_posts}")
    logger.info(f"Failed posts: {failed_posts}")

if __name__ == "__main__":
    logger.info("Script started.")
    start_time = time.time()
    process_and_post_webstories()
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Script completed. Total execution time: {total_time:.2f} seconds")

# import io
# import os
# import hashlib
# import threading
# import markdown
# import re
# import json
# import logging
# from tortoise.utils.text import split_and_recombine_text
# from flask import Flask, Response, request, jsonify, send_file, url_for
# from scipy.io.wavfile import write
# import numpy as np
# import ljinference
# import msinference
# import torch
# import yaml
# from flask_cors import CORS
# from decimal import Decimal
# import phonemizer
# from scipy.io.wavfile import write
# import io
# import re
# import torch
# from tortoise.utils.text import split_and_recombine_text
# from pydub import AudioSegment


# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def is_using_gpu():
#     return torch.cuda.is_available() and torch.cuda.current_device() >= 0

# voice_path = "voices/"

# # Load GPU config from file
# try:
#     with open('gpu_config.yml', 'r') as file:
#         gpu_config = yaml.safe_load(file)
#     gpu_device_id = gpu_config.get('gpu_device_id', 0)
# except Exception as e:
#     logger.error(f"Failed to load GPU config: {e}")
#     gpu_device_id = 999  # Default to CPU if config can't be loaded

# # Check if CUDA is available
# if torch.cuda.is_available() and gpu_device_id != 999:
#     torch.cuda.set_device(gpu_device_id)
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')

# logger.info(f"Selected device: {device}")

# def find_wav_files(directory):
#     wav_files = []
#     try:
#         files = os.listdir(directory)
#         for file in files:
#             if file.lower().endswith(".wav"):
#                 file_name_without_extension = os.path.splitext(file)[0]
#                 wav_files.append(file_name_without_extension)
#         wav_files.sort()
#     except Exception as e:
#         logger.error(f"Error finding WAV files: {e}")
#     return wav_files

# voicelist = find_wav_files(voice_path)
# voices = {}

# global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)

# logger.info("Computing voices")
# for v in voicelist:
#     try:
#         voices[v] = msinference.compute_style(f'voices/{v}.wav')
#     except Exception as e:
#         logger.error(f"Error computing style for voice {v}: {e}")

# import re


# def preprocess_tts_text(text):
#     # Ensure the text ends with proper punctuation
#     if not text.strip().endswith(('.', '!', '?')):
#         text = text.strip() + '.'
#     return text


# app = Flask(__name__)
# cors = CORS(app)

# @app.route("/")
# def index():
#     try:
#         with open('API_DOCS.md', 'r') as f:
#             return markdown.markdown(f.read())
#     except Exception as e:
#         logger.error(f"Error reading API docs: {e}")
#         return "API documentation unavailable", 500

# @app.get("/speakers")
# def get_speakers():
#     speakers_special = []
#     for speaker in voicelist:
#         preview_url = url_for('get_sample', filename=f"{speaker}.wav", _external=True)
#         speaker_special = {
#             'name': speaker,
#             'voice_id': speaker,
#             'preview_url': preview_url
#         }
#         speakers_special.append(speaker_special)
#     return jsonify(speakers_special)

# @app.get('/sample/<filename>')
# def get_sample(filename: str):
#     file_path = os.path.join(voice_path, filename)
#     if os.path.isfile(file_path):
#         return send_file(file_path, mimetype='audio/wav', as_attachment=True)
#     else:
#         logger.error(f"File not found: {file_path}")
#         return "File not found", 404


# app = Flask(__name__)
# @app.route("/api/v1/static", methods=['POST'])
# def serve_wav():
#     if 'text' not in request.form or 'voice' not in request.form:
#         error_response = {'error': 'Missing required fields. Please include "text" and "voice" in your request.'}
#         return jsonify(error_response), 400
    
#     text = request.form['text'].strip()
#     voice = request.form['voice'].strip().lower()
#     alpha_float = float(request.form.get('alpha', '0.3'))
#     beta_float = float(request.form.get('beta', '0.7'))
#     diffusion_steps_int = int(request.form.get('diffusion_steps', '15'))
#     embedding_scale_float = float(request.form.get('embedding_scale', '1'))
#     speed_float = float(request.form.get('speed', '1.0'))

#     if voice not in voices:
#         error_response = {'error': 'Invalid voice selected'}
#         return jsonify(error_response), 400
    
#     v = voices[voice]
    
#     # Split long text into chunks
#     texts = split_and_recombine_text(text)
#     audios = []
    
#     try:
#         for i, t in enumerate(texts):
#             # Preprocess the text chunk
#             processed_text = preprocess_tts_text(t)
            
#             # For all chunks except the last one, add a short pause
#             if i < len(texts) - 1:
#                 processed_text += ' '  # Add a space to create a slight pause
            
#             audio = msinference.inference(processed_text, v, alpha_float, beta_float, diffusion_steps_int, embedding_scale_float, speed_float)
#             audios.append(audio)
        
#         # Concatenate all processed chunks
#         final_audio = np.concatenate(audios)
        
#         # Add silence padding at the end
#         silence_duration_ms = 500  # 500ms of silence
#         silence_samples = int(silence_duration_ms * 24000 / 1000)  # Convert ms to samples
#         silence = np.zeros(silence_samples)
#         final_audio = np.concatenate([final_audio, silence])
        
#         output_buffer = io.BytesIO()
#         write(output_buffer, 24000, final_audio)
#         output_buffer.seek(0)
        
#         response = Response(output_buffer.getvalue())
#         response.headers["Content-Type"] = "audio/wav"
#         response.headers["X-Using-GPU"] = str(is_using_gpu())
#         return response
#     except Exception as e:
#         logger.error(f"Error generating audio: {e}")
#         return jsonify({'error': 'Failed to generate audio'}), 500