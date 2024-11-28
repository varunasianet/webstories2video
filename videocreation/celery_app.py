# import sys
# import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from app_factory import flask_app, celery
# import logging
# import redis
# from celery.signals import task_failure, worker_ready, worker_shutdown, worker_init, worker_process_init

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s')
# logger = logging.getLogger(__name__)

# # Import tasks to register them with Celery
# import tasks

# # This is important for Celery to find the app
# app = celery

# # Celery signal handlers
# @worker_init.connect
# def worker_init_handler(sender, **kwargs):
#     logger.info(f"Initializing worker {sender.hostname}")

# @worker_process_init.connect
# def worker_process_init_handler(sender, **kwargs):
#     logger.info(f"Initializing worker process")

# @worker_ready.connect
# def worker_ready_handler(sender, **kwargs):
#     logger.info(f"Worker {sender.hostname} is ready.")
#     # Test Redis connection
#     try:
#         redis_client = redis.Redis.from_url(celery.conf.broker_url)
#         redis_client.ping()
#         logger.info("Successfully connected to Redis")
#     except redis.ConnectionError:
#         logger.error("Failed to connect to Redis")

# @worker_shutdown.connect
# def worker_shutdown_handler(sender, **kwargs):
#     logger.info(f"Worker {sender.hostname} is shutting down.")

# @task_failure.connect
# def task_failure_handler(sender, task_id, exception, args, kwargs, traceback, einfo, **kw):
#     logger.error(f"Task {task_id} failed: {str(exception)}")

# if __name__ == '__main__':
#     flask_app.run(debug=False)  # Set debug to False in production

# from pydub import AudioSegment
# import os
# import requests
# import time
# import re
# import uuid
# import json
# import numpy as np
# from PIL import Image, ImageDraw, ImageFont, ImageEnhance
# from bs4 import BeautifulSoup
# from urllib.parse import urljoin
# from moviepy.editor import *
# from moviepy.audio.fx.all import volumex, audio_normalize
# from moviepy.audio.AudioClip import AudioClip
# from requests.exceptions import RequestException, Timeout, ConnectionError
# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry 
# from io import BytesIO
# from multiprocessing import Pool, cpu_count
# import cv2
# from pydub import AudioSegment
# from moviepy.audio.fx.all import audio_fadeout
# from google.cloud import storage
# import shutil
# from flask import Flask, request, jsonify
# from google.cloud import storage
# import logging
# from urllib.parse import urlparse
# from google.cloud import storage
# from google.auth.transport.requests import Request
# from google.oauth2 import service_account
# import os
# import threading
# import traceback

# MAX_CONCURRENT_VIDEOS = 3  # Default value
# video_semaphore = threading.Semaphore(MAX_CONCURRENT_VIDEOS)

# # Set the path to your service account key file
# SERVICE_ACCOUNT_FILE = '/home/varun_saagar/videocreation/creds/asianet-tech-staging-91b6f4c817e0.json'

# # Create credentials using the service account file
# credentials = service_account.Credentials.from_service_account_file(
#     SERVICE_ACCOUNT_FILE, scopes=["https://www.googleapis.com/auth/cloud-platform"]
# )

# # Create a storage client using these credentials
# storage_client = storage.Client(credentials=credentials)

# # Add this line near the top of the file, with other configurations
# BUCKET_NAME = "contentgrowth"


# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = Flask(__name__)

# def get_gcp_path(bucket_name, folder_path, file_name):
#     return f"gs://{bucket_name}/{folder_path}/{file_name}"

# def generate_signed_url(bucket_name, blob_name, expiration=3600):
#     """Generates a signed URL for a blob."""
#     storage_client = storage.Client(credentials=credentials)
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(blob_name)

#     url = blob.generate_signed_url(
#         version="v4",
#         expiration=expiration,
#         method="GET",
#         credentials=credentials
#     )

#     return url


# def copy_to_bucket(source_folder, bucket_name):
#     """
#     Copy the contents of the source folder to the specified GCS bucket,
#     maintaining the folder structure and including the unique ID under the 'English' folder.
#     """
#     print(f"Initializing storage client...")
#     bucket = storage_client.bucket(bucket_name)

#     # Get the unique ID from the source folder path
#     unique_id = os.path.basename(source_folder)

#     print(f"Starting to copy files from {source_folder} to {bucket_name}/English/{unique_id}")
#     for root, dirs, files in os.walk(source_folder):
#         for file in files:
#             local_path = os.path.join(root, file)
#             # Create the relative path starting from the unique_id folder
#             relative_path = os.path.relpath(local_path, source_folder)
#             # Construct the full path in the bucket
#             bucket_path = f"English/{unique_id}/{relative_path}"
            
#             print(f"Uploading file: {local_path} to {bucket_name}/{bucket_path}")
#             try:
#                 blob = bucket.blob(bucket_path)
#                 blob.upload_from_filename(local_path)
#                 print(f"Successfully uploaded: {local_path} to {bucket_name}/{bucket_path}")
#             except Exception as e:
#                 print(f"Error uploading {local_path}: {str(e)}")

#     print(f"Finished copying files to {bucket_name}/English/{unique_id}")

# # --- Configuration ---
# GCP_IP = "35.208.239.135"
# BASE_URL = f"http://{GCP_IP}:8001"
# TTS_URL = f"{BASE_URL}/api/v1/static"
# ERROR_LOG_FILE = "/home/varun_saagar/videocreation/error_log.txt"
# BASE_DIR = "/home/varun_saagar/videocreation"
# LOGO_PATH = "/home/varun_saagar/videocreation/artificts/newsable_logo.png"

# def create_font(styles, default_size=25, is_credit=False):
#     global GLOBAL_SCALE_FACTOR
    
#     if is_credit:
#         font_path = f"{BASE_DIR}/fonts/Verdana.ttf"
#     else:
#         font_family = styles.get('font-family', 'Roboto').split(',')[0].strip("'\"")
#         font_weight = 'Bold' if styles.get('font-weight') == 'bold' else 'Regular'
#         font_path = f"{BASE_DIR}/fonts/{font_family}/{font_family}-{font_weight}.ttf"
    
#     font_size = int(styles.get('font-size', f'{default_size}px').replace('px', ''))
#     font_size = int(font_size * GLOBAL_SCALE_FACTOR)  # Apply scaling factor
    
#     try:
#         return ImageFont.truetype(font_path, font_size)
#     except IOError:
#         print(f"Font file not found: {font_path}. Using default font.")
#         return ImageFont.load_default()

# def extract_styles(element):
#     if not element:
#         return {}
#     style = element.get('style', '')
#     styles = {}
#     for item in style.split(';'):
#         if ':' in item:
#             key, value = item.split(':', 1)
#             styles[key.strip()] = value.strip()
#     return styles

# def calculate_text_height(text, font, max_width):
#     lines = []
#     for paragraph in text.split('\n'):
#         line = []
#         for word in paragraph.split():
#             if font.getbbox(' '.join(line + [word]))[2] <= max_width:
#                 line.append(word)
#             else:
#                 lines.append(' '.join(line))
#                 line = [word]
#         lines.append(' '.join(line))
#     return sum(font.getbbox(line)[3] for line in lines)

# def parse_color(color_string):
#     if color_string.startswith('hsla'):
#         h, s, l, a = map(float, re.findall(r'[\d.]+', color_string))
#         # Convert HSLA to RGBA (simplified conversion)
#         c = (1 - abs(2 * l - 1)) * s
#         x = c * (1 - abs((h / 60) % 2 - 1))
#         m = l - c/2
#         if 0 <= h < 60:
#             r, g, b = c, x, 0
#         elif 60 <= h < 120:
#             r, g, b = x, c, 0
#         elif 120 <= h < 180:
#             r, g, b = 0, c, x
#         elif 180 <= h < 240:
#             r, g, b = 0, x, c
#         elif 240 <= h < 300:
#             r, g, b = x, 0, c
#         else:
#             r, g, b = c, 0, x
#         r, g, b = int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)
#         return (r, g, b, int(a * 255))
#     return (0, 0, 0, 255)  # Default to opaque black if parsing fails

# def calculate_text_height(text, font, max_width):
#     lines = []
#     for paragraph in text.split('\n'):
#         line = []
#         for word in paragraph.split():
#             if font.getbbox(' '.join(line + [word]))[2] <= max_width:
#                 line.append(word)
#             else:
#                 lines.append(' '.join(line))
#                 line = [word]
#         lines.append(' '.join(line))
#     return sum(font.getbbox(line)[3] for line in lines)

# # Add this near the top of your script, with other global variables
# GLOBAL_SCALE_FACTOR = 0.8 # Default scale factor

# def create_overlay(size, overall_styles, title_data, description_data, credits_data, webstories_title_data=None, is_first_slide=False):
#     global GLOBAL_SCALE_FACTOR
    
#     # Adjust overlay size based on global scale factor
#     overlay_size = (int(size[0] * GLOBAL_SCALE_FACTOR), int(size[1] * GLOBAL_SCALE_FACTOR))
#     overlay = Image.new('RGBA', overlay_size, (0, 0, 0, 0))
#     draw = ImageDraw.Draw(overlay)

#     title, title_styles = title_data
#     description, description_styles = description_data
#     image_credits, credit_styles = credits_data

#     # Adjust default font size based on global scale factor
#     default_font_size = int(60 * GLOBAL_SCALE_FACTOR)

#     # Adjust these values to change the font sizes
#     WEBSTORIES_TITLE_SIZE_MULTIPLIER = 1.2
#     TITLE_SIZE_MULTIPLIER = 1.1
#     DESCRIPTION_SIZE_MULTIPLIER = 0.9
#     CREDITS_SIZE_MULTIPLIER = 0.65

#     if webstories_title_data:
#         webstories_title, webstories_title_styles = webstories_title_data
#         webstories_title_styles['font-size'] = f'{int(default_font_size * WEBSTORIES_TITLE_SIZE_MULTIPLIER)}px'
#         webstories_title_styles['font-weight'] = 'bold'

#     if is_first_slide:
#         title_styles = description_styles.copy()
#         title_styles['font-size'] = f'{int(default_font_size * TITLE_SIZE_MULTIPLIER)}px'
#     else:
#         title_styles['font-size'] = f'{int(default_font_size * TITLE_SIZE_MULTIPLIER)}px'
#         title_styles['font-weight'] = 'bold'

#     description_styles['font-size'] = f'{int(default_font_size * DESCRIPTION_SIZE_MULTIPLIER)}px'
#     credit_styles['font-size'] = f'{int(default_font_size * CREDITS_SIZE_MULTIPLIER)}px'

#     # Adjust padding based on global scale factor
#     padding = int(60 * GLOBAL_SCALE_FACTOR)

#     max_width = overlay_size[0] - 2 * padding
    
#     if webstories_title_data:
#         webstories_title_font = create_font(webstories_title_styles, default_font_size)
#         webstories_title_height = calculate_text_height(webstories_title, webstories_title_font, max_width)
#     else:
#         webstories_title_height = 0

#     has_webstories_title = webstories_title_data is not None

#     title_font = create_font(title_styles, default_font_size)
#     description_font = create_font(description_styles, default_font_size)
#     credit_font = create_font(credit_styles, default_font_size, is_credit=True)  # Use Verdana for credits

#     title_height = calculate_text_height(title, title_font, max_width)
#     description_height = calculate_text_height(description, description_font, max_width)
#     credits_height = calculate_text_height(image_credits, credit_font, max_width)

#     # Calculate total text height
#     line_spacing = 10  # Doubled from 5
#     credits_spacing = 2  # Doubled from 1
#     total_text_height = (webstories_title_height if has_webstories_title else 0) + title_height + description_height + credits_height + 4 * padding + line_spacing + credits_spacing

#     # Adjust the overlay height calculation
#     overlay_height = min(int(overlay_size[1] * 0.4), total_text_height)

#     # Extract background color and opacity
#     background = overall_styles.get('background', '')
#     color_match = re.search(r'linear-gradient\((.*?)\)', background)
#     if color_match:
#         gradient = color_match.group(1).split(',')
#         if len(gradient) >= 2:
#             start_color = parse_color(gradient[0].strip())
#             end_color = parse_color(gradient[1].strip())
#             color = start_color  # Use start color for the overlay
#         else:
#             color = (0, 0, 0, int(0.5 * 255))  # 50% opaque black as default
#     else:
#         color = (0, 0, 0, int(0.5 * 255))  # 50% opaque black as default

#     # Create rounded rectangle overlay at the bottom of the image
#     overlay_top = overlay_size[1] - overlay_height
#     corner_radius = int(100 * GLOBAL_SCALE_FACTOR)
#     rounded_rectangle(draw, [(0, overlay_top), overlay_size], corner_radius, fill=color)

#     # Add text to the overlay
#     current_y = overlay_top + padding

#     if webstories_title_data:
#         # Draw the webstories title text centered
#         current_y = draw_text(draw, (overlay_size[0] // 2, current_y), webstories_title, webstories_title_font, max_width, color=(255, 255, 255, 255), align='center')
#         current_y += line_spacing

#     # Draw the title text centered
#     current_y = draw_text(draw, (overlay_size[0] // 2, current_y), title, title_font, max_width, color=(255, 255, 255, 255), align='center')

#     # Add line spacing
#     current_y += line_spacing

#     # Draw the description text centered
#     current_y = draw_text(draw, (overlay_size[0] // 2, current_y), description, description_font, max_width, color=(255, 255, 255, 255), align='center')

#     current_y += credits_spacing

#     # Draw credits right-aligned at the bottom
#     credits_width = credit_font.getbbox(image_credits)[2]
#     credits_height = credit_font.getbbox(image_credits)[3]
#     credits_x = overlay_size[0] - padding
#     credits_y = overlay_size[1] - padding - credits_height
#     draw_text(draw, (credits_x, credits_y), image_credits, credit_font, credits_width, color=(255, 255, 255, 255), align='right')

#     # Resize the overlay back to the original size
#     overlay = overlay.resize((int(size[0]), int(size[1])), Image.LANCZOS)

#     return ImageClip(np.array(overlay))

# def draw_text(draw, position, text, font, max_width, color=(255, 255, 255, 255), align='center'):
#     global GLOBAL_SCALE_FACTOR
#     x, y = position
#     lines = []
#     words = text.split()
#     current_line = []

#     for word in words:
#         if font.getbbox(' '.join(current_line + [word]))[2] <= max_width:
#             current_line.append(word)
#         else:
#             if current_line:
#                 lines.append(' '.join(current_line))
#                 current_line = [word]
#             else:
#                 lines.append(word)
    
#     if current_line:
#         lines.append(' '.join(current_line))

#     for line in lines:
#         line_width, line_height = font.getbbox(line)[2:4]
#         if align == 'center':
#             line_x = x - line_width // 2
#         elif align == 'right':
#             line_x = x - line_width
#         else:
#             line_x = x
#         draw.text((line_x, y), line, font=font, fill=color)
#         y += line_height + int(8 * GLOBAL_SCALE_FACTOR)  # Adjust gap between lines

#     return y

# def parse_color(color_string):
#     if color_string.startswith('hsla'):
#         h, s, l, a = map(float, re.findall(r'[\d.]+', color_string))
#         # Convert HSLA to RGBA (simplified conversion)
#         c = (1 - abs(2 * l - 1)) * s
#         x = c * (1 - abs((h / 60) % 2 - 1))
#         m = l - c/2
#         if 0 <= h < 60:
#             r, g, b = c, x, 0
#         elif 60 <= h < 120:
#             r, g, b = x, c, 0
#         elif 120 <= h < 180:
#             r, g, b = 0, c, x
#         elif 180 <= h < 240:
#             r, g, b = 0, x, c
#         elif 240 <= h < 300:
#             r, g, b = x, 0, c
#         else:
#             r, g, b = c, 0, x
#         r, g, b = int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)
#         return (r, g, b, int(a * 255))
#     return (0, 0, 0, 255)  # Default to opaque black if parsing fails

# def rounded_rectangle(draw, xy, corner_radius, fill=None, outline=None):
#     upper_left_point = xy[0]
#     bottom_right_point = xy[1]
    
#     # Increase corner radius by 50% and ensure it's an integer
#     corner_radius = int(corner_radius * 1.5)
    
#     draw.rectangle(
#         [
#             (upper_left_point[0], upper_left_point[1] + corner_radius),
#             (bottom_right_point[0], bottom_right_point[1])
#         ],
#         fill=fill,
#         outline=outline
#     )
#     draw.rectangle(
#         [
#             (upper_left_point[0] + corner_radius, upper_left_point[1]),
#             (bottom_right_point[0] - corner_radius, bottom_right_point[1])
#         ],
#         fill=fill,
#         outline=outline
#     )
#     draw.pieslice([upper_left_point, (upper_left_point[0] + corner_radius * 2, upper_left_point[1] + corner_radius * 2)],
#         180,
#         270,
#         fill=fill,
#         outline=outline
#     )
#     draw.pieslice([(bottom_right_point[0] - corner_radius * 2, upper_left_point[1]), (bottom_right_point[0], upper_left_point[1] + corner_radius * 2)],
#         270,
#         360,
#         fill=fill,
#         outline=outline
#     )

# # Add these constants
# DEFAULT_WIDTH = 1080
# DEFAULT_HEIGHT = 1920

# def crop_and_resize_image(img, target_size=(1080, 1920)):
#     width, height = img.size
#     aspect_ratio = width / height
#     target_ratio = target_size[0] / target_size[1]

#     if aspect_ratio > target_ratio:
#         # Image is wider, crop the sides
#         new_width = int(height * target_ratio)
#         left = (width - new_width) // 2
#         img = img.crop((left, 0, left + new_width, height))
#     elif aspect_ratio < target_ratio:
#         # Image is taller, crop the top and bottom
#         new_height = int(width / target_ratio)
#         top = (height - new_height) // 2
#         img = img.crop((0, top, width, top + new_height))

#     # Resize the cropped image to the target size
#     img = img.resize(target_size, Image.LANCZOS)
#     return img


# from multiprocessing import Pool, cpu_count
# def process_single_slide(slide_data):
#     i, slide, image_folder, audio_folder, tts_duration, silence_duration, zoom_factor, zoom_duration = slide_data
    
#     image_path = os.path.join(image_folder, slide['image']['local_path'])
#     if os.path.exists(image_path):
#         img = cv2.imread(image_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         overall_styles = slide['overall_styles']
        
#         if i == 0:  # First slide
#             webstories_title_data = (slide['image']['webstorie_title'], slide['title_styles'])
#             title_data = (slide['image']['title'], slide['title_styles'])
#             description_data = (slide['image']['description'], slide['description_styles'])
#         else:
#             webstories_title_data = None
#             title_data = (slide['image']['title'], slide['title_styles'])
#             description_data = (slide['image']['description'], slide['description_styles'])
        
#         credits_data = (slide['image']['credit'], slide['credit_styles'])
        
#         # Calculate slide duration based on audio length plus 1.5 seconds
#         audio_file = os.path.join(audio_folder, f"slide_{i+1}.wav")
#         if os.path.exists(audio_file):
#             audio_clip = AudioFileClip(audio_file)
#             slide_duration = audio_clip.duration + 1.5  # Add 1.5 seconds gap
#         else:
#             slide_duration = silence_duration + 1.5  # Use silence duration if no audio file

#         clip = process_image(img, overall_styles, title_data, description_data, credits_data, 
#                             duration=slide_duration, zoom_factor=zoom_factor, zoom_duration=zoom_duration,
#                             is_first_slide=(i==0), webstories_title_data=webstories_title_data)
        
#         # Add audio to the clip
#         if os.path.exists(audio_file):
#             print(f"Adding audio to slide {i+1}: {audio_file}")
#             audio_clip = AudioFileClip(audio_file)
#             # Add 1.5 seconds of silence at the end
#             silence = AudioClip(lambda t: 0, duration=1.5)
#             full_audio = CompositeAudioClip([audio_clip, silence.set_start(audio_clip.duration)])
#             clip = clip.set_audio(full_audio)
#         else:
#             print(f"Warning: Audio file not found for slide {i+1}: {audio_file}")
        
#         # Save the processed clip to a temporary file
#         temp_file = os.path.join(image_folder, f"temp_clip_{i}.mp4")
#         clip.write_videofile(temp_file, codec='libx264', audio_codec='aac', threads=2, fps=24)
#         return temp_file, slide_duration
#     else:
#         print(f"Warning: Image not found: {image_path}")
#         return None, 0

# def process_image(img, overall_styles, title_data, description_data, credits_data, duration, zoom_factor, zoom_duration, is_first_slide=False, webstories_title_data=None):
#     # Convert OpenCV image to PIL Image\global GLOBAL_SCALE_FACTOR
#     global GLOBAL_SCALE_FACTOR
#     pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
#     # Resize and crop the image to fit 1920x1080 vertical format
#     target_width, target_height = 1080, 1920  # Vertical format
#     img_width, img_height = pil_img.size
    
#     # Calculate the aspect ratio
#     aspect_ratio = img_width / img_height
#     target_ratio = target_width / target_height
    
#     if aspect_ratio > target_ratio:
#         # Image is wider, crop the sides
#         new_width = int(img_height * target_ratio)
#         left = (img_width - new_width) // 2
#         pil_img = pil_img.crop((left, 0, left + new_width, img_height))
#     else:
#         # Image is taller, crop the top and bottom
#         new_height = int(img_width / target_ratio)
#         top = (img_height - new_height) // 2
#         pil_img = pil_img.crop((0, top, img_width, top + new_height))
    
#     # Resize the cropped image to the target size
#     pil_img = pil_img.resize((target_width, target_height), Image.LANCZOS)
    
#     # Convert back to numpy array for MoviePy
#     img_array = np.array(pil_img)
    
#     # Create a background clip with the resized and cropped image
#     bg_clip = ImageClip(img_array).set_duration(duration)

#     # Apply smooth zoom effect
#     bg_clip = zoom(bg_clip, zoom_factor, zoom_duration)

#     # Create overlay clip with adjusted size
#     overlay_size = (int(target_width * GLOBAL_SCALE_FACTOR), int(target_height * GLOBAL_SCALE_FACTOR))
#     if is_first_slide and webstories_title_data:
#         overlay_clip = create_overlay(overlay_size, overall_styles, title_data, description_data, credits_data, webstories_title_data, is_first_slide=True).set_duration(duration)
#     else:
#         overlay_clip = create_overlay(overlay_size, overall_styles, title_data, description_data, credits_data, is_first_slide=is_first_slide).set_duration(duration)

#     # Resize the overlay clip back to the original size
#     overlay_clip = overlay_clip.resize((target_width, target_height))

#     # Composite the background and overlay
#     final_clip = CompositeVideoClip([bg_clip, overlay_clip]).set_duration(duration)

#     return final_clip



# def save_data(unique_id, story_data):
#     main_folder = os.path.join(BASE_DIR, 'scraped_data', unique_id)
#     os.makedirs(main_folder, exist_ok=True)

#     vertical_folder = os.path.join(main_folder, 'vertical')
#     horizontal_folder = os.path.join(main_folder, 'horizontal')
#     os.makedirs(vertical_folder, exist_ok=True)
#     os.makedirs(horizontal_folder, exist_ok=True)

#     image_folder = os.path.join(main_folder, 'Image')
#     os.makedirs(image_folder, exist_ok=True)

#     for slide in story_data:
#         img_url = slide['image'].get('url')
#         if img_url:
#             image_filename = os.path.basename(slide['image']['local_path'])
#             local_path = download_and_preprocess_image(img_url, image_folder, image_filename)
#             if local_path:
#                 slide['image']['local_path'] = local_path

#     json_path = os.path.join(main_folder, 'data.json')
#     with open(json_path, 'w', encoding='utf-8') as f:
#         json.dump(story_data, f, ensure_ascii=False, indent=2)

#     print(f"Data saved for article {unique_id}")
#     print(f"Vertical folder created: {vertical_folder}")
#     print(f"Horizontal folder created: {horizontal_folder}")

# from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

# from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
# from PIL import Image
# from io import BytesIO

# def download_and_preprocess_image(url, folder_path, filename):
#     # Parse the URL
#     parsed_url = urlparse(url)
#     query_params = parse_qs(parsed_url.query)
    
#     # Add or update the impolicy and im parameters
#     query_params['impolicy'] = ['Q-100']
#     query_params['im'] = ['Resize=(1920,1080)']
    
#     # Reconstruct the URL with the new parameters
#     new_query = urlencode(query_params, doseq=True)
#     new_url = urlunparse(parsed_url._replace(query=new_query))
    
#     response = requests.get(new_url)
#     if response.status_code == 200:
#         img = Image.open(BytesIO(response.content))
        
#         # Resize the image to 1920x1080 while maintaining aspect ratio
#         img.thumbnail((1920, 1080), Image.LANCZOS)
        
#         file_path = os.path.join(folder_path, filename)
#         img.save(file_path, 'JPEG', quality=95)
#         return file_path
#     return None



# def process_amp_story_pages(url):
#     global content_id
#     content_id = url.split('-')[-1]

#     response = requests.get(url)
#     if response.status_code != 200:
#         print(f"Failed to fetch the page: {url}")
#         return None, None

#     soup = BeautifulSoup(response.text, 'html.parser')
#     pages = soup.find_all('amp-story-page')

#     if not pages:
#         print(f"No amp-story-pages found in the URL: {url}")
#         return None, None

#     unique_id = content_id[-6:]  # Use the last 6 characters of content_id as unique_id
#     story_data = []

#     for index, page in enumerate(pages, start=1):
#         img_element = page.find('amp-img')
#         text_overlay = page.find('div', class_='text-overlay')

#         if img_element and 'src' in img_element.attrs:
#             img_url = img_element['src']
#             if not img_url.startswith('http'):
#                 img_url = urljoin(url, img_url)

#             webstorie_title = ""
#             title = ""
#             description = ""
#             credit = ""
            
#             if index == 1:  # Special handling for the first slide
#                 secname_element = text_overlay.find('p', class_='secname')
#                 title_element = text_overlay.find('h1')
#                 description_element = text_overlay.find('div', class_='description')
#                 credit_element = text_overlay.find('div', class_='credittxt')

#                 webstorie_title = secname_element.text.strip() if secname_element else ""
#                 title = title_element.text.strip() if title_element else ""
#                 description = description_element.find('p').text.strip() if description_element and description_element.find('p') else ""
#                 credit = credit_element.text.strip() if credit_element else ""
#             else:
#                 title_element = text_overlay.find(['h1', 'h2'])
#                 description_element = text_overlay.find('div', class_='description')
#                 credit_element = text_overlay.find('div', class_='credittxt')

#                 title = title_element.text.strip() if title_element else ""
#                 description = description_element.find('p').text.strip() if description_element and description_element.find('p') else ""
#                 credit = credit_element.text.strip() if credit_element else ""

#             page_data = {
#                 "slide_number": index,
#                 "image": {
#                     "webstorie_title": webstorie_title, 
#                     "title": title,
#                     "description": description,
#                     "credit": credit,
#                     "local_path": f"Image/{content_id}_image{index}.jpg",
#                     "url": img_url
#                 },
#                 "overall_styles": extract_styles(text_overlay),
#                 "title_styles": extract_styles(title_element),
#                 "description_styles": extract_styles(description_element),
#                 "credit_styles": extract_styles(credit_element),
#                 "content_id": content_id
#             }

#             story_data.append(page_data)
#     return unique_id, story_data

# def scrape_first_slide(page, base_url):
#     img = page.find('amp-img')
#     if not img:
#         return None, None, None

#     image_src = urljoin(base_url, img.get('src', ''))
#     image_alt = img.get('alt', '')
#     image_filename = "image1.jpg"

#     grid_layer = page.find('amp-story-grid-layer', class_='bottom')
#     if not grid_layer:
#         grid_layer = page  # Fallback to the entire page if the specific class is not found

#     text_overlay = grid_layer.find('div', class_='text-overlay')
#     if not text_overlay:
#         text_overlay = grid_layer  # Fallback to the grid layer if text-overlay is not found

#     # Extract the category (secname)
#     category = text_overlay.find('p', class_='secname')
#     category_content = category.text.strip() if category else ''

#     # Extract the title
#     title = text_overlay.find('h1') or text_overlay.find('h2')
#     title_content = title.text.strip() if title else ''

#     # Extract the description
#     description_div = text_overlay.find('div', class_='description')
#     description = description_div.find('p') if description_div else text_overlay.find('p')
#     description_content = description.text.strip() if description else ''

#     # Extract image credits
#     credittxt = text_overlay.find('div', class_='credittxt')
#     credit_content = credittxt.text.strip() if credittxt else ''

#     page_data = {
#         'slide_number': 1,
#         'image': {
#             'title': title_content,
#             'description': description_content,
#             'credit': credit_content,
#             'local_path': f"Image/{image_filename}"
#         },
#     }

#     return page_data, image_src, image_filename

# def scrape_other_slides(page, base_url, slide_number):
#     img = page.find('amp-img')
#     if not img:
#         return None, None, None

#     image_src = urljoin(base_url, img.get('src', ''))
#     image_alt = img.get('alt', '')
#     image_filename = f"image{slide_number}.jpg"

#     grid_layer = page.find('amp-story-grid-layer', class_='bottom')
#     if not grid_layer:
#         grid_layer = page  # Fallback to the entire page if the specific class is not found

#     text_overlay = grid_layer.find('div', class_='text-overlay')
#     if not text_overlay:
#         text_overlay = grid_layer  # Fallback to the grid layer if text-overlay is not found

#     title = text_overlay.find('h1') or text_overlay.find('h2')
#     title_content = title.text.strip() if title else ''

#     description = text_overlay.find('p')
#     description_content = description.text.strip() if description else ''

#     credittxt = text_overlay.find('div', class_='credittxt')
#     credit_content = credittxt.text.strip() if credittxt else ''

#     page_data = {
#         'slide_number': slide_number,
#         'image': {
#             'title': title_content,
#             'description': description_content,
#             'credit': credit_content,
#             'local_path': f"Image/{image_filename}"
#         },
#     }

#     return page_data, image_src, image_filename

# def custom_resize(clip, width=None, height=None):
#     """Resize a clip to a given width or height, maintaining aspect ratio."""
#     if width is None and height is None:
#         raise ValueError("Either width or height must be provided")

#     aspect_ratio = clip.w / clip.h
#     if width is not None:
#         height = int(width / aspect_ratio)
#     elif height is not None:
#         width = int(height * aspect_ratio)
#     return clip.resize((width, height))



# def zoom(clip, zoom_factor, zoom_duration):
#     def zoom_effect(get_frame, t):
#         t = t % zoom_duration
#         current_zoom = 1 + (zoom_factor - 1) * t / zoom_duration
#         frame = get_frame(t)
#         h, w = frame.shape[:2]
#         zoomed_frame = cv2.resize(frame, None, fx=current_zoom, fy=current_zoom, interpolation=cv2.INTER_LINEAR)
#         zh, zw = zoomed_frame.shape[:2]
#         y1 = int((zh - h) / 2)
#         x1 = int((zw - w) / 2)
#         return zoomed_frame[y1:y1+h, x1:x1+w]
#     return clip.fl(zoom_effect)

# def generate_tts_parallel(slide_data):
#     i, tts_text, audio_folder = slide_data
#     tts_output_path = os.path.join(audio_folder, f"slide_{i+1}.wav")
#     success, audio_duration = generate_tts(tts_text, tts_output_path)
    
#     if success:
#         print(f"TTS generated for slide {i+1} (duration: {audio_duration:.2f}s)")
#     else:
#         print(f"Failed to generate TTS for slide {i+1}")
#         audio_duration = 0
    
#     return audio_duration

# def generate_tts(text, output_path, voice="f-us-2", max_retries=5, retry_backoff=1):
#     data = {
#         "text": text,
#         "voice": voice,
#         "alpha": "0.4",
#         "beta": "0.7",
#         "diffusion_steps": "15",
#         "embedding_scale": "1",
#         "speed": "1.1"
#     }

#     session = requests.Session()
#     retry_strategy = Retry(
#         total=max_retries,
#         backoff_factor=retry_backoff,
#         status_forcelist=[500, 502, 503, 504],
#         allowed_methods=["POST"]
#     )
#     adapter = HTTPAdapter(max_retries=retry_strategy)
#     session.mount("http://", adapter)

#     try:
#         response = session.post(TTS_URL, data=data, timeout=300)
#         response.raise_for_status()
#         with open(output_path, "wb") as f:
#             f.write(response.content)
#         return True, AudioFileClip(output_path).duration
#     except Exception as e:
#         print(f"TTS request failed: {e}")
#         return False, 0


# from PIL import Image
# import numpy as np

# def add_logo(clip, logo_path=LOGO_PATH, opacity=0.75, duration=None, size_factor=2.0, padding=40):
#     global GLOBAL_SCALE_FACTOR
#     """
#     Adds a logo to the top-right corner of a video clip, with size control, background removal, and padding.
    
#     :param padding: Number of pixels to pad the logo from the top and right edges.
#     """
#     if not os.path.exists(logo_path):
#         print(f"Warning: Logo file not found: {logo_path}")
#         return clip

#     # Open the logo image
#     logo_img = Image.open(logo_path).convert("RGBA")
#     logo_array = np.array(logo_img)

#     # Create a mask for non-black and non-transparent pixels
#     mask = (logo_array[:,:,0] > 10) | (logo_array[:,:,1] > 10) | (logo_array[:,:,2] > 10) | (logo_array[:,:,3] > 0)

#     # Find the bounding box of the non-black area
#     rows = np.any(mask, axis=1)
#     cols = np.any(mask, axis=0)
#     ymin, ymax = np.where(rows)[0][[0, -1]]
#     xmin, xmax = np.where(cols)[0][[0, -1]]

#     # Crop the logo to remove the black margin
#     cropped_logo = logo_img.crop((xmin, ymin, xmax+1, ymax+1))

#     # Calculate new dimensions while maintaining aspect ratio
#     original_width, original_height = cropped_logo.size
#     new_width = int(original_width * size_factor * GLOBAL_SCALE_FACTOR)
#     new_height = int(original_height * size_factor * GLOBAL_SCALE_FACTOR)

#     # Adjust padding
#     adjusted_padding = int(padding * GLOBAL_SCALE_FACTOR)

#     # Create the logo clip
#     logo = (
#         ImageClip(np.array(cropped_logo))
#         .resize((new_width, new_height))
#         .set_duration(duration or clip.duration)
#         .set_opacity(opacity)
#         .set_position(lambda t: (clip.w - new_width - adjusted_padding, adjusted_padding))
#     )

#     return CompositeVideoClip([clip, logo])


# def update_global_scale_factor(new_scale_factor):
#     global GLOBAL_SCALE_FACTOR
#     GLOBAL_SCALE_FACTOR = new_scale_factor
#     print(f"Updated GLOBAL_SCALE_FACTOR to {GLOBAL_SCALE_FACTOR}")

# def create_video(image_folder, output_video_path, target_duration, overlay_clips=None, bgm_path=None, bgm_volume=None, audio_folder=None, silence_duration=3, output_format='default', intro_path=None, end_credits_path=None):
#     image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))])

#     clips = []
#     audio_clips = []
    
#     # Add intro video if provided
#     if intro_path and os.path.exists(intro_path):
#         intro_clip = VideoFileClip(intro_path)
#         intro_duration = intro_clip.duration
#         clips.append(intro_clip)
#         audio_clips.append(intro_clip.audio)
#     else:
#         intro_duration = 0
    
#     for i, image_file in enumerate(image_files):
#         image_path = os.path.join(image_folder, image_file)
        
#         if overlay_clips and i < len(overlay_clips):
#             clip = overlay_clips[i]
#         else:
#             clip = ImageClip(image_path)
        
#         if audio_folder:
#             audio_file = os.path.join(audio_folder, f"slide_{i+1}.wav")
#             if os.path.exists(audio_file):
#                 # Load the audio file using pydub
#                 slide_audio = AudioSegment.from_wav(audio_file)
                
#                 # Add silence at the beginning
#                 silence = AudioSegment.silent(duration=silence_duration * 1000)
#                 full_audio = silence + slide_audio
                
#                 # Apply fade out
#                 fade_duration = min(500, len(full_audio))  # 500ms fade out, or shorter if the audio is very short
#                 full_audio = full_audio.fade_out(duration=fade_duration)
                
#                 # Export the processed audio
#                 processed_audio_file = os.path.join(audio_folder, f"processed_slide_{i+1}.wav")
#                 full_audio.export(processed_audio_file, format="wav")
                
#                 # Use the processed audio in the video clip
#                 clip_audio = AudioFileClip(processed_audio_file)
#                 clip = clip.set_audio(clip_audio)
#                 clip_duration = len(full_audio) / 1000.0  # Duration in seconds
#                 clip = clip.set_duration(clip_duration)
#                 audio_clips.append(clip_audio)
#             else:
#                 # If there's no audio, just use the clip with silence duration
#                 clip = clip.set_duration(silence_duration)
#                 audio_clips.append(AudioClip(lambda t: 0, duration=silence_duration))
#         clips.append(clip)

#     # Add end credits video if provided
#     if end_credits_path and os.path.exists(end_credits_path):
#         end_credits_clip = VideoFileClip(end_credits_path)
#         end_credits_duration = end_credits_clip.duration
#         clips.append(end_credits_clip)
#         audio_clips.append(end_credits_clip.audio)
#     else:
#         end_credits_duration = 0

#     final_clip = concatenate_videoclips(clips, method="compose")
#     final_clip = add_logo(final_clip)  # Add the logo here after compositing

#     # Combine all audio clips with crossfade
#     crossfade_duration = 0.1  # 100ms crossfade
#     final_audio = CompositeAudioClip([audio_clips[0]])
#     for i in range(1, len(audio_clips)):
#         final_audio = CompositeAudioClip([
#             final_audio,
#             audio_clips[i].set_start(final_audio.duration - crossfade_duration)
#         ])

#     # Apply fade-out to the final audio
#     final_fade_duration = 1  # 1 second fade-out at the end
#     final_audio = final_audio.audio_fadeout(final_fade_duration)

#     # Add background music
#     if bgm_path and os.path.exists(bgm_path):
#         background_audio = AudioFileClip(bgm_path)
#         if background_audio.duration < final_clip.duration:
#             background_audio = afx.audio_loop(background_audio, duration=final_clip.duration)
#         else:
#             background_audio = background_audio.subclip(0, final_clip.duration)
        
#         # Apply fade-out to background music
#         background_audio = background_audio.audio_fadeout(final_fade_duration)
        
#         # Mix the final audio with the background music
#         final_audio = CompositeAudioClip([final_audio, background_audio.volumex(bgm_volume)])

#     # Set the final audio to the video
#     final_clip = final_clip.set_audio(final_audio)

#     # Set audio codec to AAC and video codec to H.264 with faster preset
#     final_clip.write_videofile(output_video_path, fps=24, codec='libx264', audio_codec='aac', audio_bitrate='128k',
#                                preset='faster', threads=cpu_count())

#     return intro_duration, end_credits_duration

# def convert_vertical_to_horizontal(input_path, output_path):
#     # Load the vertical video
#     clip = VideoFileClip(input_path)
    
#     # Define the target resolution (e.g., 1920x1080 for Full HD)
#     target_width = 1920
#     target_height = 1080
    
#     # Calculate the scaling factor to fit the height
#     scale_factor = target_height / clip.h
    
#     # Resize the clip while maintaining aspect ratio
#     resized_clip = clip.resize(height=target_height)
    
#     # Create a black background
#     background = ColorClip(size=(target_width, target_height), color=(0,0,0))
#     background = background.set_duration(clip.duration)
    
#     # Calculate the position to center the video
#     x_center = (target_width - resized_clip.w) / 2
    
#     # Composite the resized clip onto the background
#     final_clip = CompositeVideoClip([background, resized_clip.set_position((x_center, 0))])
    
#     # Write the final horizontal video
#     final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=24)  # Add fps=24 here
    
#     # Close the clips
#     clip.close()
#     final_clip.close()
    
# def process_slide(slide, i, image_folder, audio_folder, tts_duration, silence_duration, zoom_factor, zoom_duration):
#     image_path = os.path.join(image_folder, slide['image']['local_path'])
#     if os.path.exists(image_path):
#         img = cv2.imread(image_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         overall_styles = slide['overall_styles']
        
#         if i == 0:  # First slide
#             webstories_title_data = (slide['image']['webstorie_title'], slide['title_styles'])
#             title_data = (slide['image']['title'], slide['title_styles'])
#             description_data = (slide['image']['description'], slide['description_styles'])
#         else:
#             webstories_title_data = None
#             title_data = (slide['image']['title'], slide['title_styles'])
#             description_data = (slide['image']['description'], slide['description_styles'])
        
#         credits_data = (slide['image']['credit'], slide['credit_styles'])
        
#         slide_duration = max(tts_duration + silence_duration, zoom_duration)
#         clip = process_image(img, overall_styles, title_data, description_data, credits_data, 
#                             duration=slide_duration, zoom_factor=zoom_factor, zoom_duration=zoom_duration,
#                             is_first_slide=(i==0), webstories_title_data=webstories_title_data)
        
#         return clip, slide_duration
#     else:
#         print(f"Warning: Image not found: {image_path}")
#         return None, 0
# def create_final_video(temp_clip_files, output_path, bgm_path, bgm_volume, intro_path, end_credits_path):
#     clips = []
#     for temp_file in temp_clip_files:
#         clip = VideoFileClip(temp_file)
#         clips.append(clip)

#     if intro_path and os.path.exists(intro_path):
#         intro_clip = VideoFileClip(intro_path)
#         clips.insert(0, intro_clip)

#     if end_credits_path and os.path.exists(end_credits_path):
#         end_credits_clip = VideoFileClip(end_credits_path)
#         clips.append(end_credits_clip)

#     final_clip = concatenate_videoclips(clips, method="compose")
#     final_clip = add_logo(final_clip)

#     # Add background music if provided
#     if bgm_path and os.path.exists(bgm_path):
#         background_audio = AudioFileClip(bgm_path)
#         if background_audio.duration < final_clip.duration:
#             background_audio = afx.audio_loop(background_audio, duration=final_clip.duration)
#         else:
#             background_audio = background_audio.subclip(0, final_clip.duration)
        
#         background_audio = background_audio.volumex(bgm_volume)
        
#         # Mix the original audio with the background music
#         final_audio = CompositeAudioClip([final_clip.audio, background_audio])
#         final_clip = final_clip.set_audio(final_audio)

#     final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', threads=cpu_count(), fps=24)
#     return final_clip.duration


# def main(url, bgm_path, zoom_factor, zoom_duration, bgm_volume, silence_duration, use_tts=True, output_format='vertical', intro_path=None, end_credits_path=None):    
#     unique_id = None
#     main_folder = None
#     vertical_gcp_path = None
#     horizontal_gcp_path = None

#     try:
#         print("Step 1: Processing AMP story pages...")
#         result = process_amp_story_pages(url)
#         if not result:
#             raise ValueError("Failed to scrape data from the URL")

#         unique_id, story_data = result
#         content_id = unique_id

#         for slide in story_data:
#             slide['content_id'] = content_id

#         print("Step 2: Saving scraped data...")
#         save_data(unique_id, story_data)
#         print("All data has been scraped and saved.")

#         base_dir = BASE_DIR
#         main_folder = os.path.join(base_dir, 'scraped_data', unique_id)
#         vertical_folder = os.path.join(main_folder, 'vertical')
#         horizontal_folder = os.path.join(main_folder, 'horizontal')

#         os.makedirs(vertical_folder, exist_ok=True)
#         os.makedirs(horizontal_folder, exist_ok=True)

#         image_folder = os.path.join(main_folder, 'Image')
#         audio_folder = os.path.join(main_folder, 'Audio')
#         os.makedirs(audio_folder, exist_ok=True)

#         print("Step 3: Generating TTS...")
#         if use_tts:
#             tts_data = []
#             for i, slide in enumerate(story_data):
#                 if i == 0:
#                     tts_text = f"{slide['image']['webstorie_title']}. {slide['image']['title']}. {slide['image']['description']}"
#                 else:
#                     tts_text = f"{slide['image']['title']}. {slide['image']['description']}"
#                 tts_data.append((i, tts_text, audio_folder))

#             with Pool(processes=cpu_count()) as pool:
#                 tts_durations = pool.map(generate_tts_parallel, tts_data)
#         else:
#             tts_durations = [0] * len(story_data)

#         print("Step 4: Processing slides in parallel...")
#         slide_data = []
#         for i, slide in enumerate(story_data):
#             slide_data.append((i, slide, image_folder, audio_folder, tts_durations[i], silence_duration, zoom_factor, zoom_duration))

#         with Pool(processes=cpu_count()) as pool:
#             results = pool.map(process_single_slide, slide_data)

#         temp_clip_files = [result[0] for result in results if result[0]]
#         slide_durations = [result[1] for result in results if result[0]]

#         print("Step 5: Creating vertical video...")
#         output_path = os.path.join(vertical_folder, f'{content_id}_vertical.mp4')
#         video_duration = create_final_video(temp_clip_files, output_path, bgm_path, bgm_volume, intro_path, end_credits_path)

#         print("Step 6: Converting vertical to horizontal...")
#         if output_path and os.path.exists(output_path):
#             horizontal_output_path = os.path.join(horizontal_folder, f'{content_id}_horizontal.mp4')
#             try:
#                 logger.info(f"Starting horizontal conversion for {content_id}")
#                 convert_vertical_to_horizontal(output_path, horizontal_output_path)
                
#                 if os.path.exists(horizontal_output_path):
#                     file_size = os.path.getsize(horizontal_output_path)
#                     logger.info(f"Horizontal video created: {horizontal_output_path} (Size: {file_size} bytes)")
#                 else:
#                     raise FileNotFoundError(f"Horizontal video file not found: {horizontal_output_path}")
                
#             except Exception as e:
#                 logger.error(f"Error creating horizontal video for {content_id}: {str(e)}", exc_info=True)
#                 log_error(f"Horizontal video creation error for {content_id}: {str(e)}")
#         else:
#             logger.warning(f"Vertical video not found for {content_id}, skipping horizontal conversion")

#         print("Step 7: Copying files to the 'contentgrowth' bucket...")
#         if main_folder:
#             try:
#                 copy_to_bucket(main_folder, 'contentgrowth')
#                 print(f"All files from {main_folder} have been copied to the 'contentgrowth/English' bucket.")
                
#                 # Generate correct GCP paths
#                 vertical_gcp_path = f"English/{unique_id}/vertical/{content_id}_vertical.mp4"
#                 horizontal_gcp_path = f"English/{unique_id}/horizontal/{content_id}_horizontal.mp4"
                
#             except Exception as e:
#                 print(f"An error occurred while copying files to the bucket: {str(e)}")
#                 log_error("Bucket copy error", str(e))
#         else:
#             print("Main folder not created, skipping bucket copy.")

#         print("Step 8: Removing local files...")
#         try:
#             if main_folder and os.path.exists(main_folder):
#                 shutil.rmtree(main_folder)
#                 print(f"Local files in {main_folder} have been removed.")
#             else:
#                 print("Main folder not found, skipping local file removal.")
#         except Exception as e:
#             print(f"An error occurred while removing local files: {str(e)}")
#             log_error("Local file removal error", str(e))

#         # Clean up temporary files
#         for temp_file in temp_clip_files:
#             try:
#                 os.remove(temp_file)
#             except Exception as e:
#                 print(f"Error removing temporary file {temp_file}: {str(e)}")

#         return vertical_gcp_path, horizontal_gcp_path

#     except ValueError as ve:
#         print(f"Validation error: {ve}")
#         if unique_id:
#             log_error(f"Article {unique_id}: {ve}")
#         raise
#     except Exception as e:
#         print(f"Unexpected error: {e}")
#         traceback.print_exc()
#         if unique_id:
#             log_error(f"Article {unique_id}: {e}", traceback.format_exc())
#         raise

# @app.route('/create_video', methods=['POST'])
# def create_video_api():
#     url = request.json.get('url')
#     if not url:
#         return jsonify({"error": "URL is required"}), 400

#     # Validate URL
#     try:
#         result = urlparse(url)
#         if not all([result.scheme, result.netloc]):
#             return jsonify({"error": "Invalid URL"}), 400
#     except ValueError:
#         return jsonify({"error": "Invalid URL"}), 400

#     # Get parameters from request or use defaults
#     bgm_path = request.json.get('bgm_path', f"{BASE_DIR}/Background_music/happy/07augsug.mp3")
#     zoom_factor = request.json.get('zoom_factor', 1.15)
#     zoom_duration = request.json.get('zoom_duration', 20)
#     bgm_volume = request.json.get('bgm_volume', 0.15)
#     silence_duration = request.json.get('silence_duration', 1)
#     use_tts = request.json.get('use_tts', True)
#     intro_path = request.json.get('intro_path', "/home/varun_saagar/videocreation/artificts/intro-fixed.mp4")
#     end_credits_path = request.json.get('end_credits_path', "/home/varun_saagar/videocreation/artificts/Newsable end sting__Vertical.mp4")
#     # Get scale factor from request or use default
#     scale_factor = request.json.get('scale_factor', 1.0)
#     update_global_scale_factor(scale_factor)


#     try:
#         start_time = time.time()
#         vertical_gcp_path, horizontal_gcp_path = main(
#             url, bgm_path, zoom_factor, zoom_duration, bgm_volume, silence_duration, use_tts,
#             output_format='vertical', intro_path=intro_path, end_credits_path=end_credits_path
#         )
#         end_time = time.time()
#         total_time = end_time - start_time

#         if vertical_gcp_path and horizontal_gcp_path:
#             logger.info(f"Video creation successful. Execution time: {total_time:.2f} seconds")
#             return jsonify({
#                 "vertical_video_url": vertical_gcp_path,
#                 "horizontal_video_url": horizontal_gcp_path,
#                 "execution_time": f"{total_time:.2f} seconds"
#             }), 200
#         else:
#             logger.error("Failed to generate video URLs")
#             return jsonify({"error": "Failed to generate video URLs"}), 500

#     except ValueError as ve:
#         logger.error(f"Validation error: {str(ve)}")
#         return jsonify({"error": str(ve)}), 400
#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}", exc_info=True)
#         return jsonify({"error": "An unexpected error occurred"}), 500

# @app.route('/health', methods=['GET'])
# def health_check():
#     return jsonify({"status": "healthy"}), 200

# @app.route('/update_concurrency', methods=['POST'])
# def update_concurrency():
#     new_value = request.json.get('max_concurrent_videos')
#     if new_value is None:
#         logger.error("max_concurrent_videos is required")
#         return jsonify({"error": "max_concurrent_videos is required"}), 400
    
#     try:
#         new_value = int(new_value)
#         if new_value <= 0:
#             logger.error("max_concurrent_videos must be a positive integer")
#             return jsonify({"error": "max_concurrent_videos must be a positive integer"}), 400
        
#         update_max_concurrent_videos(new_value)
#         logger.info(f"MAX_CONCURRENT_VIDEOS updated to {new_value}")
#         return jsonify({"message": f"MAX_CONCURRENT_VIDEOS updated to {new_value}"}), 200
#     except ValueError:
#         logger.error("max_concurrent_videos must be a valid integer")
#         return jsonify({"error": "max_concurrent_videos must be a valid integer"}), 400

# def update_max_concurrent_videos(new_value):
#     global MAX_CONCURRENT_VIDEOS
#     global video_semaphore
    
#     MAX_CONCURRENT_VIDEOS = new_value
#     video_semaphore = threading.Semaphore(MAX_CONCURRENT_VIDEOS)
#     logger.info(f"Updated MAX_CONCURRENT_VIDEOS to {MAX_CONCURRENT_VIDEOS}")

# def log_error(message, details=None):
#     log_file = ERROR_LOG_FILE
#     with open(log_file, "a") as f:
#         f.write(f"\n{message}\n")
#         if details:
#             f.write(details)


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=6000, debug=False)

from pydub import AudioSegment
import os
import requests
import time
import re
import uuid
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from moviepy.editor import *
from moviepy.audio.fx.all import volumex, audio_normalize
from moviepy.audio.AudioClip import AudioClip
from requests.exceptions import RequestException, Timeout, ConnectionError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry 
from io import BytesIO
from multiprocessing import Pool, cpu_count
import cv2
from pydub import AudioSegment
from moviepy.audio.fx.all import audio_fadeout
from google.cloud import storage
import shutil
from flask import Flask, request, jsonify
from google.cloud import storage
import logging
from urllib.parse import urlparse
from google.cloud import storage
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import os
import threading
import traceback

MAX_CONCURRENT_VIDEOS = 3  # Default value
video_semaphore = threading.Semaphore(MAX_CONCURRENT_VIDEOS)

# Set the path to your service account key file
SERVICE_ACCOUNT_FILE = '/home/varun_saagar/videocreation/creds/asianet-tech-staging-91b6f4c817e0.json'

# Create credentials using the service account file
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

# Create a storage client using these credentials
storage_client = storage.Client(credentials=credentials)

# Add this line near the top of the file, with other configurations
BUCKET_NAME = "contentgrowth"


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def get_gcp_path(bucket_name, folder_path, file_name):
    return f"gs://{bucket_name}/{folder_path}/{file_name}"

def generate_signed_url(bucket_name, blob_name, expiration=3600):
    """Generates a signed URL for a blob."""
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    url = blob.generate_signed_url(
        version="v4",
        expiration=expiration,
        method="GET",
        credentials=credentials
    )

    return url


def copy_to_bucket(source_folder, bucket_name):
    """
    Copy the contents of the source folder to the specified GCS bucket,
    maintaining the folder structure and including the unique ID under the 'English' folder.
    """
    print(f"Initializing storage client...")
    bucket = storage_client.bucket(bucket_name)

    # Get the unique ID from the source folder path
    unique_id = os.path.basename(source_folder)

    print(f"Starting to copy files from {source_folder} to {bucket_name}/English/{unique_id}")
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            local_path = os.path.join(root, file)
            # Create the relative path starting from the unique_id folder
            relative_path = os.path.relpath(local_path, source_folder)
            # Construct the full path in the bucket
            bucket_path = f"English/{unique_id}/{relative_path}"
            
            print(f"Uploading file: {local_path} to {bucket_name}/{bucket_path}")
            try:
                blob = bucket.blob(bucket_path)
                blob.upload_from_filename(local_path)
                print(f"Successfully uploaded: {local_path} to {bucket_name}/{bucket_path}")
            except Exception as e:
                print(f"Error uploading {local_path}: {str(e)}")

    print(f"Finished copying files to {bucket_name}/English/{unique_id}")

# --- Configuration ---
GCP_IP = "35.208.239.135"
BASE_URL = f"http://{GCP_IP}:8001"
TTS_URL = f"{BASE_URL}/api/v1/static"
ERROR_LOG_FILE = "/home/varun_saagar/videocreation/error_log.txt"
BASE_DIR = "/home/varun_saagar/videocreation"
LOGO_PATH = "/home/varun_saagar/videocreation/artificts/newsable_logo.png"

def create_font(styles, default_size=25, is_credit=False):
    if is_credit:
        font_path = f"{BASE_DIR}/fonts/Verdana.ttf"
    else:
        font_family = styles.get('font-family', 'Roboto').split(',')[0].strip("'\"")
        font_weight = 'Bold' if styles.get('font-weight') == 'bold' else 'Regular'
        font_path = f"{BASE_DIR}/fonts/{font_family}/{font_family}-{font_weight}.ttf"
    
    font_size = int(styles.get('font-size', f'{default_size}px').replace('px', ''))
    
    try:
        return ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Font file not found: {font_path}. Using default font.")
        return ImageFont.load_default()

def extract_styles(element):
    if not element:
        return {}
    style = element.get('style', '')
    styles = {}
    for item in style.split(';'):
        if ':' in item:
            key, value = item.split(':', 1)
            styles[key.strip()] = value.strip()
    return styles

def calculate_text_height(text, font, max_width):
    lines = []
    for paragraph in text.split('\n'):
        line = []
        for word in paragraph.split():
            if font.getbbox(' '.join(line + [word]))[2] <= max_width:
                line.append(word)
            else:
                lines.append(' '.join(line))
                line = [word]
        lines.append(' '.join(line))
    return sum(font.getbbox(line)[3] for line in lines)

def parse_color(color_string):
    if color_string.startswith('hsla'):
        h, s, l, a = map(float, re.findall(r'[\d.]+', color_string))
        # Convert HSLA to RGBA (simplified conversion)
        c = (1 - abs(2 * l - 1)) * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = l - c/2
        if 0 <= h < 60:
            r, g, b = c, x, 0
        elif 60 <= h < 120:
            r, g, b = x, c, 0
        elif 120 <= h < 180:
            r, g, b = 0, c, x
        elif 180 <= h < 240:
            r, g, b = 0, x, c
        elif 240 <= h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        r, g, b = int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)
        return (r, g, b, int(a * 255))
    return (0, 0, 0, 255)  # Default to opaque black if parsing fails

def calculate_text_height(text, font, max_width):
    lines = []
    for paragraph in text.split('\n'):
        line = []
        for word in paragraph.split():
            if font.getbbox(' '.join(line + [word]))[2] <= max_width:
                line.append(word)
            else:
                lines.append(' '.join(line))
                line = [word]
        lines.append(' '.join(line))
    return sum(font.getbbox(line)[3] for line in lines)

def create_overlay(size, overall_styles, title_data, description_data, credits_data, webstories_title_data=None, is_first_slide=False):
    overlay_size = (size[0] * 2, size[1] * 2)  # Double the size of the overlay
    overlay = Image.new('RGBA', overlay_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    title, title_styles = title_data
    description, description_styles = description_data
    image_credits, credit_styles = credits_data

    # Set default styles
    default_font_size = 40  # Doubled from 20

    # Adjust these values to change the font sizes
    WEBSTORIES_TITLE_SIZE_MULTIPLIER = 1.2
    TITLE_SIZE_MULTIPLIER = 1.1
    DESCRIPTION_SIZE_MULTIPLIER = 0.9
    CREDITS_SIZE_MULTIPLIER = 0.65

    if webstories_title_data:
        webstories_title, webstories_title_styles = webstories_title_data
        webstories_title_styles['font-size'] = f'{int(default_font_size * WEBSTORIES_TITLE_SIZE_MULTIPLIER)}px'
        webstories_title_styles['font-weight'] = 'bold'

    if is_first_slide:
        title_styles = description_styles.copy()
        title_styles['font-size'] = f'{int(default_font_size * TITLE_SIZE_MULTIPLIER)}px'
    else:
        title_styles['font-size'] = f'{int(default_font_size * TITLE_SIZE_MULTIPLIER)}px'
        title_styles['font-weight'] = 'bold'

    description_styles['font-size'] = f'{int(default_font_size * DESCRIPTION_SIZE_MULTIPLIER)}px'
    credit_styles['font-size'] = f'{int(default_font_size * CREDITS_SIZE_MULTIPLIER)}px'

    # Calculate text heights
    padding = 40  # Doubled from 20
    max_width = overlay_size[0] - 2 * padding
    
    if webstories_title_data:
        webstories_title_font = create_font(webstories_title_styles, default_font_size)
        webstories_title_height = calculate_text_height(webstories_title, webstories_title_font, max_width)
    else:
        webstories_title_height = 0

    has_webstories_title = webstories_title_data is not None

    title_font = create_font(title_styles, default_font_size)
    description_font = create_font(description_styles, default_font_size)
    credit_font = create_font(credit_styles, default_font_size, is_credit=True)  # Use Verdana for credits

    title_height = calculate_text_height(title, title_font, max_width)
    description_height = calculate_text_height(description, description_font, max_width)
    credits_height = calculate_text_height(image_credits, credit_font, max_width)

    # Calculate total text height
    line_spacing = 10  # Doubled from 5
    credits_spacing = 2  # Doubled from 1
    total_text_height = (webstories_title_height if has_webstories_title else 0) + title_height + description_height + credits_height + 4 * padding + line_spacing + credits_spacing

    # Adjust the overlay height calculation
    overlay_height = min(int(overlay_size[1] * 0.4), total_text_height)

    # Extract background color and opacity
    background = overall_styles.get('background', '')
    color_match = re.search(r'linear-gradient\((.*?)\)', background)
    if color_match:
        gradient = color_match.group(1).split(',')
        if len(gradient) >= 2:
            start_color = parse_color(gradient[0].strip())
            end_color = parse_color(gradient[1].strip())
            color = start_color  # Use start color for the overlay
        else:
            color = (0, 0, 0, int(0.5 * 255))  # 50% opaque black as default
    else:
        color = (0, 0, 0, int(0.5 * 255))  # 50% opaque black as default

    # Create rounded rectangle overlay at the bottom of the image
    overlay_top = overlay_size[1] - overlay_height
    corner_radius = 100  # Doubled from 50
    rounded_rectangle(draw, [(0, overlay_top), overlay_size], corner_radius, fill=color)

    # Add text to the overlay
    current_y = overlay_top + padding

    if webstories_title_data:
        # Draw the webstories title text centered
        current_y = draw_text(draw, (overlay_size[0] // 2, current_y), webstories_title, webstories_title_font, max_width, color=(255, 255, 255, 255), align='center')
        current_y += line_spacing

    # Draw the title text centered
    current_y = draw_text(draw, (overlay_size[0] // 2, current_y), title, title_font, max_width, color=(255, 255, 255, 255), align='center')

    # Add line spacing
    current_y += line_spacing

    # Draw the description text centered
    current_y = draw_text(draw, (overlay_size[0] // 2, current_y), description, description_font, max_width, color=(255, 255, 255, 255), align='center')

    current_y += credits_spacing

    # Draw credits right-aligned at the bottom
    credits_width = credit_font.getbbox(image_credits)[2]
    credits_height = credit_font.getbbox(image_credits)[3]
    credits_x = overlay_size[0] - padding
    credits_y = overlay_size[1] - padding - credits_height
    draw_text(draw, (credits_x, credits_y), image_credits, credit_font, credits_width, color=(255, 255, 255, 255), align='right')

    # Resize the overlay back to the original size
    overlay = overlay.resize(size, Image.LANCZOS)

    return ImageClip(np.array(overlay))

def draw_text(draw, position, text, font, max_width, color=(255, 255, 255, 255), align='center'):
    x, y = position
    lines = []
    words = text.split()
    current_line = []

    for word in words:
        if font.getbbox(' '.join(current_line + [word]))[2] <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                lines.append(word)
    
    if current_line:
        lines.append(' '.join(current_line))

    for line in lines:
        line_width, line_height = font.getbbox(line)[2:4]
        if align == 'center':
            line_x = x - line_width // 2
        elif align == 'right':
            line_x = x - line_width
        else:
            line_x = x
        draw.text((line_x, y), line, font=font, fill=color, antialias=True)
        y += line_height + 5  # Add a small gap between lines

    return y  # Return the new y position after drawing all lines

def parse_color(color_string):
    if color_string.startswith('hsla'):
        h, s, l, a = map(float, re.findall(r'[\d.]+', color_string))
        # Convert HSLA to RGBA (simplified conversion)
        c = (1 - abs(2 * l - 1)) * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = l - c/2
        if 0 <= h < 60:
            r, g, b = c, x, 0
        elif 60 <= h < 120:
            r, g, b = x, c, 0
        elif 120 <= h < 180:
            r, g, b = 0, c, x
        elif 180 <= h < 240:
            r, g, b = 0, x, c
        elif 240 <= h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        r, g, b = int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)
        return (r, g, b, int(a * 255))
    return (0, 0, 0, 255)  # Default to opaque black if parsing fails

def rounded_rectangle(draw, xy, corner_radius, fill=None, outline=None):
    upper_left_point = xy[0]
    bottom_right_point = xy[1]
    
    draw.rectangle(
        [
            (upper_left_point[0], upper_left_point[1] + corner_radius),
            (bottom_right_point[0], bottom_right_point[1])
        ],
        fill=fill,
        outline=outline
    )
    draw.rectangle(
        [
            (upper_left_point[0] + corner_radius, upper_left_point[1]),
            (bottom_right_point[0] - corner_radius, bottom_right_point[1])
        ],
        fill=fill,
        outline=outline
    )
    draw.pieslice([upper_left_point, (upper_left_point[0] + corner_radius * 2, upper_left_point[1] + corner_radius * 2)],
        180,
        270,
        fill=fill,
        outline=outline
    )
    draw.pieslice([(bottom_right_point[0] - corner_radius * 2, upper_left_point[1]), (bottom_right_point[0], upper_left_point[1] + corner_radius * 2)],
        270,
        360,
        fill=fill,
        outline=outline
    )

# Add these constants
DEFAULT_WIDTH = 1080
DEFAULT_HEIGHT = 1920

def crop_and_resize_image(img, target_size=(1080, 1920)):
    width, height = img.size
    aspect_ratio = width / height
    target_ratio = target_size[0] / target_size[1]

    if aspect_ratio > target_ratio:
        # Image is wider, crop the sides
        new_width = int(height * target_ratio)
        left = (width - new_width) // 2
        img = img.crop((left, 0, left + new_width, height))
    elif aspect_ratio < target_ratio:
        # Image is taller, crop the top and bottom
        new_height = int(width / target_ratio)
        top = (height - new_height) // 2
        img = img.crop((0, top, width, top + new_height))

    # Resize the cropped image to the target size
    img = img.resize(target_size, Image.LANCZOS)
    return img


from multiprocessing import Pool, cpu_count
def process_single_slide(slide_data):
    i, slide, image_folder, audio_folder, tts_duration, silence_duration, zoom_factor, zoom_duration = slide_data
    
    image_path = os.path.join(image_folder, slide['image']['local_path'])
    if os.path.exists(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Read image with alpha channel if present
        
        overall_styles = slide['overall_styles']
        
        if i == 0:  # First slide
            webstories_title_data = (slide['image']['webstorie_title'], slide['title_styles'])
            title_data = (slide['image']['title'], slide['title_styles'])
            description_data = (slide['image']['description'], slide['description_styles'])
        else:
            webstories_title_data = None
            title_data = (slide['image']['title'], slide['title_styles'])
            description_data = (slide['image']['description'], slide['description_styles'])
        
        credits_data = (slide['image']['credit'], slide['credit_styles'])
        
        # Calculate slide duration based on audio length plus 1.5 seconds
        audio_file = os.path.join(audio_folder, f"slide_{i+1}.wav")
        if os.path.exists(audio_file):
            audio_clip = AudioFileClip(audio_file)
            slide_duration = audio_clip.duration + 1.5  # Add 1.5 seconds gap
        else:
            slide_duration = silence_duration + 1.5  # Use silence duration if no audio file

        clip = process_image(img, overall_styles, title_data, description_data, credits_data, 
                            duration=slide_duration, zoom_factor=zoom_factor, zoom_duration=zoom_duration,
                            is_first_slide=(i==0), webstories_title_data=webstories_title_data)
        
        # Add audio to the clip
        if os.path.exists(audio_file):
            print(f"Adding audio to slide {i+1}: {audio_file}")
            audio_clip = AudioFileClip(audio_file)
            # Add 1.5 seconds of silence at the end
            silence = AudioClip(lambda t: 0, duration=1.5)
            full_audio = CompositeAudioClip([audio_clip, silence.set_start(audio_clip.duration)])
            clip = clip.set_audio(full_audio)
        else:
            print(f"Warning: Audio file not found for slide {i+1}: {audio_file}")
        
        # Save the processed clip to a temporary file
        temp_file = os.path.join(image_folder, f"temp_clip_{i}.mp4")
        clip.write_videofile(temp_file, codec='libx264', audio_codec='aac', threads=2, fps=24)
        return temp_file, slide_duration
    else:
        print(f"Warning: Image not found: {image_path}")
        return None, 0

def process_image(img, overall_styles, title_data, description_data, credits_data, duration, zoom_factor, zoom_duration, is_first_slide=False, webstories_title_data=None):
    # Convert OpenCV image to PIL Image without color space conversion
    pil_img = Image.fromarray(img)
    
    # Resize and crop the image to fit 1080x1920 vertical format
    target_width, target_height = 1080, 1920  # Vertical format
    img_width, img_height = pil_img.size
    
    # Calculate the aspect ratio
    aspect_ratio = img_width / img_height
    target_ratio = target_width / target_height
    
    if aspect_ratio > target_ratio:
        # Image is wider, crop the sides
        new_width = int(img_height * target_ratio)
        left = (img_width - new_width) // 2
        pil_img = pil_img.crop((left, 0, left + new_width, img_height))
    else:
        # Image is taller, crop the top and bottom
        new_height = int(img_width / target_ratio)
        top = (img_height - new_height) // 2
        pil_img = pil_img.crop((0, top, img_width, top + new_height))
    
    # Resize the cropped image to the target size
    pil_img = pil_img.resize((target_width, target_height), Image.LANCZOS)
    
    # Convert back to numpy array for MoviePy
    img_array = np.array(pil_img)
    
    # Create a background clip with the resized and cropped image
    bg_clip = ImageClip(img_array).set_duration(duration)

    # Apply smooth zoom effect
    bg_clip = zoom(bg_clip, zoom_factor, zoom_duration)

    # Create overlay clip
    if is_first_slide and webstories_title_data:
        overlay_clip = create_overlay((target_width, target_height), overall_styles, title_data, description_data, credits_data, webstories_title_data, is_first_slide=True).set_duration(duration)
    else:
        overlay_clip = create_overlay((target_width, target_height), overall_styles, title_data, description_data, credits_data, is_first_slide=is_first_slide).set_duration(duration)

    # Composite the background and overlay
    final_clip = CompositeVideoClip([bg_clip, overlay_clip]).set_duration(duration)

    return final_clip


def save_data(unique_id, story_data):
    main_folder = os.path.join(BASE_DIR, 'scraped_data', unique_id)
    os.makedirs(main_folder, exist_ok=True)

    vertical_folder = os.path.join(main_folder, 'vertical')
    horizontal_folder = os.path.join(main_folder, 'horizontal')
    os.makedirs(vertical_folder, exist_ok=True)
    os.makedirs(horizontal_folder, exist_ok=True)

    image_folder = os.path.join(main_folder, 'Image')
    os.makedirs(image_folder, exist_ok=True)

    for slide in story_data:
        img_url = slide['image'].get('url')
        if img_url:
            image_filename = os.path.basename(slide['image']['local_path'])
            local_path = download_and_preprocess_image(img_url, image_folder, image_filename)
            if local_path:
                slide['image']['local_path'] = local_path

    json_path = os.path.join(main_folder, 'data.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(story_data, f, ensure_ascii=False, indent=2)

    print(f"Data saved for article {unique_id}")
    print(f"Vertical folder created: {vertical_folder}")
    print(f"Horizontal folder created: {horizontal_folder}")

from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
from PIL import Image
from io import BytesIO

def download_and_preprocess_image(url, folder_path, filename):
    # Parse the URL
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    
    # Add or update the impolicy and im parameters
    query_params['impolicy'] = ['Q-100']
    query_params['im'] = ['Resize=(1920,1080)']
    
    # Reconstruct the URL with the new parameters
    new_query = urlencode(query_params, doseq=True)
    new_url = urlunparse(parsed_url._replace(query=new_query))
    
    response = requests.get(new_url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        
        # Resize the image to 1920x1080 while maintaining aspect ratio
        img.thumbnail((1920, 1080), Image.LANCZOS)
        
        file_path = os.path.join(folder_path, filename)
        img.save(file_path, 'JPEG', quality=95)
        return file_path
    return None



def process_amp_story_pages(url):
    global content_id
    content_id = url.split('-')[-1]

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch the page: {url}")
        return None, None

    soup = BeautifulSoup(response.text, 'html.parser')
    pages = soup.find_all('amp-story-page')

    if not pages:
        print(f"No amp-story-pages found in the URL: {url}")
        return None, None

    unique_id = content_id[-6:]  # Use the last 6 characters of content_id as unique_id
    story_data = []

    for index, page in enumerate(pages, start=1):
        img_element = page.find('amp-img')
        text_overlay = page.find('div', class_='text-overlay')

        if img_element and 'src' in img_element.attrs:
            img_url = img_element['src']
            if not img_url.startswith('http'):
                img_url = urljoin(url, img_url)

            webstorie_title = ""
            title = ""
            description = ""
            credit = ""
            
            if index == 1:  # Special handling for the first slide
                secname_element = text_overlay.find('p', class_='secname')
                title_element = text_overlay.find('h1')
                description_element = text_overlay.find('div', class_='description')
                credit_element = text_overlay.find('div', class_='credittxt')

                webstorie_title = secname_element.text.strip() if secname_element else ""
                title = title_element.text.strip() if title_element else ""
                description = description_element.find('p').text.strip() if description_element and description_element.find('p') else ""
                credit = credit_element.text.strip() if credit_element else ""
            else:
                title_element = text_overlay.find(['h1', 'h2'])
                description_element = text_overlay.find('div', class_='description')
                credit_element = text_overlay.find('div', class_='credittxt')

                title = title_element.text.strip() if title_element else ""
                description = description_element.find('p').text.strip() if description_element and description_element.find('p') else ""
                credit = credit_element.text.strip() if credit_element else ""

            page_data = {
                "slide_number": index,
                "image": {
                    "webstorie_title": webstorie_title, 
                    "title": title,
                    "description": description,
                    "credit": credit,
                    "local_path": f"Image/{content_id}_image{index}.jpg",
                    "url": img_url
                },
                "overall_styles": extract_styles(text_overlay),
                "title_styles": extract_styles(title_element),
                "description_styles": extract_styles(description_element),
                "credit_styles": extract_styles(credit_element),
                "content_id": content_id
            }

            story_data.append(page_data)
    return unique_id, story_data

def scrape_first_slide(page, base_url):
    img = page.find('amp-img')
    if not img:
        return None, None, None

    image_src = urljoin(base_url, img.get('src', ''))
    image_alt = img.get('alt', '')
    image_filename = "image1.jpg"

    grid_layer = page.find('amp-story-grid-layer', class_='bottom')
    if not grid_layer:
        grid_layer = page  # Fallback to the entire page if the specific class is not found

    text_overlay = grid_layer.find('div', class_='text-overlay')
    if not text_overlay:
        text_overlay = grid_layer  # Fallback to the grid layer if text-overlay is not found

    # Extract the category (secname)
    category = text_overlay.find('p', class_='secname')
    category_content = category.text.strip() if category else ''

    # Extract the title
    title = text_overlay.find('h1') or text_overlay.find('h2')
    title_content = title.text.strip() if title else ''

    # Extract the description
    description_div = text_overlay.find('div', class_='description')
    description = description_div.find('p') if description_div else text_overlay.find('p')
    description_content = description.text.strip() if description else ''

    # Extract image credits
    credittxt = text_overlay.find('div', class_='credittxt')
    credit_content = credittxt.text.strip() if credittxt else ''

    page_data = {
        'slide_number': 1,
        'image': {
            'title': title_content,
            'description': description_content,
            'credit': credit_content,
            'local_path': f"Image/{image_filename}"
        },
    }

    return page_data, image_src, image_filename

def scrape_other_slides(page, base_url, slide_number):
    img = page.find('amp-img')
    if not img:
        return None, None, None

    image_src = urljoin(base_url, img.get('src', ''))
    image_alt = img.get('alt', '')
    image_filename = f"image{slide_number}.jpg"

    grid_layer = page.find('amp-story-grid-layer', class_='bottom')
    if not grid_layer:
        grid_layer = page  # Fallback to the entire page if the specific class is not found

    text_overlay = grid_layer.find('div', class_='text-overlay')
    if not text_overlay:
        text_overlay = grid_layer  # Fallback to the grid layer if text-overlay is not found

    title = text_overlay.find('h1') or text_overlay.find('h2')
    title_content = title.text.strip() if title else ''

    description = text_overlay.find('p')
    description_content = description.text.strip() if description else ''

    credittxt = text_overlay.find('div', class_='credittxt')
    credit_content = credittxt.text.strip() if credittxt else ''

    page_data = {
        'slide_number': slide_number,
        'image': {
            'title': title_content,
            'description': description_content,
            'credit': credit_content,
            'local_path': f"Image/{image_filename}"
        },
    }

    return page_data, image_src, image_filename

def custom_resize(clip, width=None, height=None):
    """Resize a clip to a given width or height, maintaining aspect ratio."""
    if width is None and height is None:
        raise ValueError("Either width or height must be provided")

    aspect_ratio = clip.w / clip.h
    if width is not None:
        height = int(width / aspect_ratio)
    elif height is not None:
        width = int(height * aspect_ratio)
    return clip.resize((width, height))



def zoom(clip, zoom_factor, zoom_duration):
    def zoom_effect(get_frame, t):
        t = t % zoom_duration
        current_zoom = 1 + (zoom_factor - 1) * t / zoom_duration
        frame = get_frame(t)
        h, w = frame.shape[:2]
        zoomed_frame = cv2.resize(frame, None, fx=current_zoom, fy=current_zoom, interpolation=cv2.INTER_LINEAR)
        zh, zw = zoomed_frame.shape[:2]
        y1 = int((zh - h) / 2)
        x1 = int((zw - w) / 2)
        return zoomed_frame[y1:y1+h, x1:x1+w]
    return clip.fl(zoom_effect)

def generate_tts_parallel(slide_data):
    i, tts_text, audio_folder = slide_data
    tts_output_path = os.path.join(audio_folder, f"slide_{i+1}.wav")
    success, audio_duration = generate_tts(tts_text, tts_output_path)
    
    if success:
        print(f"TTS generated for slide {i+1} (duration: {audio_duration:.2f}s)")
    else:
        print(f"Failed to generate TTS for slide {i+1}")
        audio_duration = 0
    
    return audio_duration

def generate_tts(text, output_path, voice="f-us-2", max_retries=5, retry_backoff=1):
    data = {
        "text": text,
        "voice": voice,
        "alpha": "0.4",
        "beta": "0.7",
        "diffusion_steps": "15",
        "embedding_scale": "1",
        "speed": "1.1"
    }

    session = requests.Session()
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=retry_backoff,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)

    try:
        response = session.post(TTS_URL, data=data, timeout=300)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(response.content)
        return True, AudioFileClip(output_path).duration
    except Exception as e:
        print(f"TTS request failed: {e}")
        return False, 0


from PIL import Image
import numpy as np

def add_logo(clip, logo_path=LOGO_PATH, opacity=0.75, duration=None, size_factor=2.0, padding=40):
    """
    Adds a logo to the top-right corner of a video clip, with size control, background removal, and padding.
    
    :param padding: Number of pixels to pad the logo from the top and right edges.
    """
    if not os.path.exists(logo_path):
        print(f"Warning: Logo file not found: {logo_path}")
        return clip

    # Open the logo image
    logo_img = Image.open(logo_path).convert("RGBA")
    logo_array = np.array(logo_img)

    # Create a mask for non-black and non-transparent pixels
    mask = (logo_array[:,:,0] > 10) | (logo_array[:,:,1] > 10) | (logo_array[:,:,2] > 10) | (logo_array[:,:,3] > 0)

    # Find the bounding box of the non-black area
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    # Crop the logo to remove the black margin
    cropped_logo = logo_img.crop((xmin, ymin, xmax+1, ymax+1))

    # Calculate new dimensions while maintaining aspect ratio
    original_width, original_height = cropped_logo.size
    new_width = int(original_width * size_factor)
    new_height = int(original_height * size_factor)

    # Create the logo clip
    logo = (
        ImageClip(np.array(cropped_logo))
        .resize((new_width, new_height))
        .set_duration(duration or clip.duration)
        .set_opacity(opacity)
        .set_position(lambda t: (clip.w - new_width - padding, padding))  # Dynamic positioning with padding
    )

    return CompositeVideoClip([clip, logo])

def create_video(image_folder, output_video_path, target_duration, overlay_clips=None, bgm_path=None, bgm_volume=None, audio_folder=None, silence_duration=3, output_format='default', intro_path=None, end_credits_path=None):
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))])

    clips = []
    audio_clips = []
    
    # Add intro video if provided
    if intro_path and os.path.exists(intro_path):
        intro_clip = VideoFileClip(intro_path)
        intro_duration = intro_clip.duration
        clips.append(intro_clip)
        audio_clips.append(intro_clip.audio)
    else:
        intro_duration = 0
    
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        
        if overlay_clips and i < len(overlay_clips):
            clip = overlay_clips[i]
        else:
            clip = ImageClip(image_path)
        
        if audio_folder:
            audio_file = os.path.join(audio_folder, f"slide_{i+1}.wav")
            if os.path.exists(audio_file):
                # Load the audio file using pydub
                slide_audio = AudioSegment.from_wav(audio_file)
                
                # Add silence at the beginning
                silence = AudioSegment.silent(duration=silence_duration * 1000)
                full_audio = silence + slide_audio
                
                # Apply fade out
                fade_duration = min(500, len(full_audio))  # 500ms fade out, or shorter if the audio is very short
                full_audio = full_audio.fade_out(duration=fade_duration)
                
                # Export the processed audio
                processed_audio_file = os.path.join(audio_folder, f"processed_slide_{i+1}.wav")
                full_audio.export(processed_audio_file, format="wav")
                
                # Use the processed audio in the video clip
                clip_audio = AudioFileClip(processed_audio_file)
                clip = clip.set_audio(clip_audio)
                clip_duration = len(full_audio) / 1000.0  # Duration in seconds
                clip = clip.set_duration(clip_duration)
                audio_clips.append(clip_audio)
            else:
                # If there's no audio, just use the clip with silence duration
                clip = clip.set_duration(silence_duration)
                audio_clips.append(AudioClip(lambda t: 0, duration=silence_duration))
        clips.append(clip)

    # Add end credits video if provided
    if end_credits_path and os.path.exists(end_credits_path):
        end_credits_clip = VideoFileClip(end_credits_path)
        end_credits_duration = end_credits_clip.duration
        clips.append(end_credits_clip)
        audio_clips.append(end_credits_clip.audio)
    else:
        end_credits_duration = 0

    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip = add_logo(final_clip)  # Add the logo here after compositing

    # Combine all audio clips with crossfade
    crossfade_duration = 0.1  # 100ms crossfade
    final_audio = CompositeAudioClip([audio_clips[0]])
    for i in range(1, len(audio_clips)):
        final_audio = CompositeAudioClip([
            final_audio,
            audio_clips[i].set_start(final_audio.duration - crossfade_duration)
        ])

    # Apply fade-out to the final audio
    final_fade_duration = 1  # 1 second fade-out at the end
    final_audio = final_audio.audio_fadeout(final_fade_duration)

    # Add background music
    if bgm_path and os.path.exists(bgm_path):
        background_audio = AudioFileClip(bgm_path)
        if background_audio.duration < final_clip.duration:
            background_audio = afx.audio_loop(background_audio, duration=final_clip.duration)
        else:
            background_audio = background_audio.subclip(0, final_clip.duration)
        
        # Apply fade-out to background music
        background_audio = background_audio.audio_fadeout(final_fade_duration)
        
        # Mix the final audio with the background music
        final_audio = CompositeAudioClip([final_audio, background_audio.volumex(bgm_volume)])

    # Set the final audio to the video
    final_clip = final_clip.set_audio(final_audio)

    # Set audio codec to AAC and video codec to H.264 with faster preset
    final_clip.write_videofile(output_video_path, fps=24, codec='libx264', audio_codec='aac', audio_bitrate='128k',
                               preset='faster', threads=cpu_count())

    return intro_duration, end_credits_duration

def convert_vertical_to_horizontal(input_path, output_path):
    # Load the vertical video
    clip = VideoFileClip(input_path)
    
    # Define the target resolution (e.g., 1920x1080 for Full HD)
    target_width = 1920
    target_height = 1080
    
    # Calculate the scaling factor to fit the height
    scale_factor = target_height / clip.h
    
    # Resize the clip while maintaining aspect ratio
    resized_clip = clip.resize(height=target_height)
    
    # Create a black background
    background = ColorClip(size=(target_width, target_height), color=(0,0,0))
    background = background.set_duration(clip.duration)
    
    # Calculate the position to center the video
    x_center = (target_width - resized_clip.w) / 2
    
    # Composite the resized clip onto the background
    final_clip = CompositeVideoClip([background, resized_clip.set_position((x_center, 0))])
    
    # Write the final horizontal video
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=24)  # Add fps=24 here
    
    # Close the clips
    clip.close()
    final_clip.close()
    
def process_slide(slide, i, image_folder, audio_folder, tts_duration, silence_duration, zoom_factor, zoom_duration):
    image_path = os.path.join(image_folder, slide['image']['local_path'])
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        overall_styles = slide['overall_styles']
        
        if i == 0:  # First slide
            webstories_title_data = (slide['image']['webstorie_title'], slide['title_styles'])
            title_data = (slide['image']['title'], slide['title_styles'])
            description_data = (slide['image']['description'], slide['description_styles'])
        else:
            webstories_title_data = None
            title_data = (slide['image']['title'], slide['title_styles'])
            description_data = (slide['image']['description'], slide['description_styles'])
        
        credits_data = (slide['image']['credit'], slide['credit_styles'])
        
        slide_duration = max(tts_duration + silence_duration, zoom_duration)
        clip = process_image(img, overall_styles, title_data, description_data, credits_data, 
                            duration=slide_duration, zoom_factor=zoom_factor, zoom_duration=zoom_duration,
                            is_first_slide=(i==0), webstories_title_data=webstories_title_data)
        
        return clip, slide_duration
    else:
        print(f"Warning: Image not found: {image_path}")
        return None, 0
def create_final_video(temp_clip_files, output_path, bgm_path, bgm_volume, intro_path, end_credits_path):
    clips = []
    for temp_file in temp_clip_files:
        clip = VideoFileClip(temp_file)
        clips.append(clip)

    if intro_path and os.path.exists(intro_path):
        intro_clip = VideoFileClip(intro_path)
        clips.insert(0, intro_clip)

    if end_credits_path and os.path.exists(end_credits_path):
        end_credits_clip = VideoFileClip(end_credits_path)
        clips.append(end_credits_clip)

    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip = add_logo(final_clip)

    # Add background music if provided
    if bgm_path and os.path.exists(bgm_path):
        background_audio = AudioFileClip(bgm_path)
        if background_audio.duration < final_clip.duration:
            background_audio = afx.audio_loop(background_audio, duration=final_clip.duration)
        else:
            background_audio = background_audio.subclip(0, final_clip.duration)
        
        background_audio = background_audio.volumex(bgm_volume)
        
        # Mix the original audio with the background music
        final_audio = CompositeAudioClip([final_clip.audio, background_audio])
        final_clip = final_clip.set_audio(final_audio)

    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', threads=cpu_count(), fps=24)
    return final_clip.duration


def main(url, bgm_path, zoom_factor, zoom_duration, bgm_volume, silence_duration, use_tts=True, output_format='vertical', intro_path=None, end_credits_path=None):    
    unique_id = None
    main_folder = None
    vertical_gcp_path = None
    horizontal_gcp_path = None

    try:
        print("Step 1: Processing AMP story pages...")
        result = process_amp_story_pages(url)
        if not result:
            raise ValueError("Failed to scrape data from the URL")

        unique_id, story_data = result
        content_id = unique_id

        for slide in story_data:
            slide['content_id'] = content_id

        print("Step 2: Saving scraped data...")
        save_data(unique_id, story_data)
        print("All data has been scraped and saved.")

        base_dir = BASE_DIR
        main_folder = os.path.join(base_dir, 'scraped_data', unique_id)
        vertical_folder = os.path.join(main_folder, 'vertical')
        horizontal_folder = os.path.join(main_folder, 'horizontal')

        os.makedirs(vertical_folder, exist_ok=True)
        os.makedirs(horizontal_folder, exist_ok=True)

        image_folder = os.path.join(main_folder, 'Image')
        audio_folder = os.path.join(main_folder, 'Audio')
        os.makedirs(audio_folder, exist_ok=True)

        print("Step 3: Generating TTS...")
        if use_tts:
            tts_data = []
            for i, slide in enumerate(story_data):
                if i == 0:
                    tts_text = f"{slide['image']['webstorie_title']}. {slide['image']['title']}. {slide['image']['description']}"
                else:
                    tts_text = f"{slide['image']['title']}. {slide['image']['description']}"
                tts_data.append((i, tts_text, audio_folder))

            with Pool(processes=cpu_count()) as pool:
                tts_durations = pool.map(generate_tts_parallel, tts_data)
        else:
            tts_durations = [0] * len(story_data)

        print("Step 4: Processing slides in parallel...")
        slide_data = []
        for i, slide in enumerate(story_data):
            slide_data.append((i, slide, image_folder, audio_folder, tts_durations[i], silence_duration, zoom_factor, zoom_duration))

        with Pool(processes=cpu_count()) as pool:
            results = pool.map(process_single_slide, slide_data)

        temp_clip_files = [result[0] for result in results if result[0]]
        slide_durations = [result[1] for result in results if result[0]]

        print("Step 5: Creating vertical video...")
        output_path = os.path.join(vertical_folder, f'{content_id}_vertical.mp4')
        video_duration = create_final_video(temp_clip_files, output_path, bgm_path, bgm_volume, intro_path, end_credits_path)

        print("Step 6: Converting vertical to horizontal...")
        if output_path and os.path.exists(output_path):
            horizontal_output_path = os.path.join(horizontal_folder, f'{content_id}_horizontal.mp4')
            try:
                logger.info(f"Starting horizontal conversion for {content_id}")
                convert_vertical_to_horizontal(output_path, horizontal_output_path)
                
                if os.path.exists(horizontal_output_path):
                    file_size = os.path.getsize(horizontal_output_path)
                    logger.info(f"Horizontal video created: {horizontal_output_path} (Size: {file_size} bytes)")
                else:
                    raise FileNotFoundError(f"Horizontal video file not found: {horizontal_output_path}")
                
            except Exception as e:
                logger.error(f"Error creating horizontal video for {content_id}: {str(e)}", exc_info=True)
                log_error(f"Horizontal video creation error for {content_id}: {str(e)}")
        else:
            logger.warning(f"Vertical video not found for {content_id}, skipping horizontal conversion")

        print("Step 7: Copying files to the 'contentgrowth' bucket...")
        if main_folder:
            try:
                copy_to_bucket(main_folder, 'contentgrowth')
                print(f"All files from {main_folder} have been copied to the 'contentgrowth/English' bucket.")
                
                # Generate correct GCP paths
                vertical_gcp_path = f"English/{unique_id}/vertical/{content_id}_vertical.mp4"
                horizontal_gcp_path = f"English/{unique_id}/horizontal/{content_id}_horizontal.mp4"
                
            except Exception as e:
                print(f"An error occurred while copying files to the bucket: {str(e)}")
                log_error("Bucket copy error", str(e))
        else:
            print("Main folder not created, skipping bucket copy.")

        print("Step 8: Removing local files...")
        try:
            if main_folder and os.path.exists(main_folder):
                shutil.rmtree(main_folder)
                print(f"Local files in {main_folder} have been removed.")
            else:
                print("Main folder not found, skipping local file removal.")
        except Exception as e:
            print(f"An error occurred while removing local files: {str(e)}")
            log_error("Local file removal error", str(e))

        # Clean up temporary files
        for temp_file in temp_clip_files:
            try:
                os.remove(temp_file)
            except Exception as e:
                print(f"Error removing temporary file {temp_file}: {str(e)}")

        return vertical_gcp_path, horizontal_gcp_path

    except ValueError as ve:
        print(f"Validation error: {ve}")
        if unique_id:
            log_error(f"Article {unique_id}: {ve}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        if unique_id:
            log_error(f"Article {unique_id}: {e}", traceback.format_exc())
        raise

@app.route('/create_video', methods=['POST'])
def create_video_api():
    url = request.json.get('url')
    if not url:
        return jsonify({"error": "URL is required"}), 400

    # Validate URL
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            return jsonify({"error": "Invalid URL"}), 400
    except ValueError:
        return jsonify({"error": "Invalid URL"}), 400

    # Get parameters from request or use defaults
    bgm_path = request.json.get('bgm_path', f"{BASE_DIR}/Background_music/happy/07augsug.mp3")
    zoom_factor = request.json.get('zoom_factor', 1.15)
    zoom_duration = request.json.get('zoom_duration', 20)
    bgm_volume = request.json.get('bgm_volume', 0.15)
    silence_duration = request.json.get('silence_duration', 1)
    use_tts = request.json.get('use_tts', True)
    intro_path = request.json.get('intro_path', "/home/varun_saagar/videocreation/artificts/intro-fixed.mp4")
    end_credits_path = request.json.get('end_credits_path', "/home/varun_saagar/videocreation/artificts/Newsable end sting__Vertical.mp4")

    try:
        start_time = time.time()
        vertical_gcp_path, horizontal_gcp_path = main(
            url, bgm_path, zoom_factor, zoom_duration, bgm_volume, silence_duration, use_tts,
            output_format='vertical', intro_path=intro_path, end_credits_path=end_credits_path
        )
        end_time = time.time()
        total_time = end_time - start_time

        if vertical_gcp_path and horizontal_gcp_path:
            logger.info(f"Video creation successful. Execution time: {total_time:.2f} seconds")
            return jsonify({
                "vertical_video_url": vertical_gcp_path,
                "horizontal_video_url": horizontal_gcp_path,
                "execution_time": f"{total_time:.2f} seconds"
            }), 200
        else:
            logger.error("Failed to generate video URLs")
            return jsonify({"error": "Failed to generate video URLs"}), 500

    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/update_concurrency', methods=['POST'])
def update_concurrency():
    new_value = request.json.get('max_concurrent_videos')
    if new_value is None:
        logger.error("max_concurrent_videos is required")
        return jsonify({"error": "max_concurrent_videos is required"}), 400
    
    try:
        new_value = int(new_value)
        if new_value <= 0:
            logger.error("max_concurrent_videos must be a positive integer")
            return jsonify({"error": "max_concurrent_videos must be a positive integer"}), 400
        
        update_max_concurrent_videos(new_value)
        logger.info(f"MAX_CONCURRENT_VIDEOS updated to {new_value}")
        return jsonify({"message": f"MAX_CONCURRENT_VIDEOS updated to {new_value}"}), 200
    except ValueError:
        logger.error("max_concurrent_videos must be a valid integer")
        return jsonify({"error": "max_concurrent_videos must be a valid integer"}), 400

def update_max_concurrent_videos(new_value):
    global MAX_CONCURRENT_VIDEOS
    global video_semaphore
    
    MAX_CONCURRENT_VIDEOS = new_value
    video_semaphore = threading.Semaphore(MAX_CONCURRENT_VIDEOS)
    logger.info(f"Updated MAX_CONCURRENT_VIDEOS to {MAX_CONCURRENT_VIDEOS}")

def log_error(message, details=None):
    log_file = ERROR_LOG_FILE
    with open(log_file, "a") as f:
        f.write(f"\n{message}\n")
        if details:
            f.write(details)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

