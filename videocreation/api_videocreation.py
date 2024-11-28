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
from celery import Celery
import time
from app_factory import flask_app, celery

# from tasks import create_video_task
import sys

print(f"Executing api_videocreation.py. __name__ is: {__name__}")
print(f"sys.modules keys: {list(sys.modules.keys())}")

app = flask_app


# Set the path to your service account key file
SERVICE_ACCOUNT_FILE = '/home/varun_saagar/videocreation/creds/asianet-tech-staging-91b6f4c817e0.json'

# Create credentials using the service account file
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

# Create a storage client using these credentials
storage_client = storage.Client(credentials=credentials)


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_gcp_path(bucket_name, folder_path, file_name):
    return f"gs://{bucket_name}/{folder_path}/{file_name}"

def generate_signed_url(bucket_name, blob_name, expiration=3600):
    """Generates a signed URL for a blob."""
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
    # Calculate the scale factor based on the new image size
    scale_factor = size[1] / 720  # Assuming the original height was 1920

    overlay_size = size
    overlay = Image.new('RGBA', overlay_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    title, title_styles = title_data
    description, description_styles = description_data
    image_credits, credit_styles = credits_data

    # Set default styles and scale them
    default_font_size = int(2.5 * 8 * scale_factor)

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
    padding = int(20 * scale_factor)
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
    line_spacing = 5 * scale_factor  # Adjust this value to increase/decrease space between title and description
    credits_spacing = 1.2 * scale_factor  # New variable for spacing before credits (reduced from 5)
    total_text_height = (webstories_title_height if has_webstories_title else 0) + title_height + description_height + credits_height + 4 * padding + line_spacing + credits_spacing

    if webstories_title_data:
        webstories_title_font = create_font(webstories_title_styles, default_font_size)
        webstories_title_height = calculate_text_height(webstories_title, webstories_title_font, max_width)
    else:
        webstories_title_height = 0

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

    # Extract border radius (only for top corners)
    border_radius = overall_styles.get('border-radius', '50px 50px 0 0')  # Default to 50px for top corners
    # Create rounded rectangle overlay at the bottom of the image
    overlay_top = overlay_size[1] - overlay_height
    corner_radius = int(50 * scale_factor)  # Scale the corner radius
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
        draw.text((line_x, y), line, font=font, fill=color)
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

def crop_and_resize_image(img, target_size=(DEFAULT_WIDTH, DEFAULT_HEIGHT)):
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

def process_image(img, overall_styles, title_data, description_data, credits_data, duration, zoom_factor, zoom_duration, is_first_slide=False, webstories_title_data=None):
    # Resize the image using OpenCV
    img = cv2.resize(img, (DEFAULT_WIDTH, DEFAULT_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    
    # Convert OpenCV image to PIL Image for overlay creation
    pil_img = Image.fromarray(img)
    
    bg_clip = ImageClip(img).set_duration(duration)

    # Apply smooth zoom effect
    bg_clip = zoom(bg_clip, zoom_factor, zoom_duration)

    if is_first_slide and webstories_title_data:
        overlay_clip = create_overlay((DEFAULT_WIDTH, DEFAULT_HEIGHT), overall_styles, title_data, description_data, credits_data, webstories_title_data, is_first_slide=True).set_duration(duration)
    else:
        overlay_clip = create_overlay((DEFAULT_WIDTH, DEFAULT_HEIGHT), overall_styles, title_data, description_data, credits_data, is_first_slide=is_first_slide).set_duration(duration)

    final_clip = CompositeVideoClip([bg_clip, overlay_clip], size=(DEFAULT_WIDTH, DEFAULT_HEIGHT)).set_duration(duration)

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


def download_and_preprocess_image(url, folder_path, filename):
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        img = crop_and_resize_image(img, (DEFAULT_WIDTH, DEFAULT_HEIGHT))
        file_path = os.path.join(folder_path, filename)
        img.save(file_path, 'JPEG')
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
    """
    Smoothly zooms in a video clip over its duration, repeating the animation.
    
    :param clip: The video clip to apply the zoom effect to
    :param zoom_factor: The maximum zoom factor
    :param zoom_duration: The duration of one zoom cycle in seconds
    """
    def zoom_effect(get_frame, t):
        # Repeat the animation every zoom_duration seconds
        t = t % zoom_duration
        current_zoom = 1 + (zoom_factor - 1) * t / zoom_duration
        frame = get_frame(t)
        zoomed_frame = custom_resize(ImageClip(frame), height=int(frame.shape[0] * current_zoom))
        return zoomed_frame.get_frame(0)
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
import re

def preprocess_tts_text(text):
    # Ensure the text ends with proper punctuation
    if not text.strip().endswith(('.', '!', '?')):
        text = text.strip() + '.'
    
    # Add padding tokens at the end
    text += ' ...'  # Adding ellipsis as padding
    
    return text

def generate_tts(text, output_path, voice="f-us-2", max_retries=5, retry_backoff=1):
    # Preprocess the text
    processed_text = preprocess_tts_text(text)
    
    data = {
        "text": processed_text,
        "voice": voice,
        "alpha": "0.4",
        "beta": "0.7",
        "diffusion_steps": "15",
        "embedding_scale": "1",
        "speed": "1.1"
    }

    # Session with retry logic
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
        
        # Save the audio
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        # Apply fade out to the audio
        audio = AudioSegment.from_wav(output_path)
        fade_duration = min(500, len(audio))  # 500ms fade out, or shorter if the audio is very short
        audio = audio.fade_out(duration=fade_duration)
        audio.export(output_path, format="wav")
        
        return True, AudioFileClip(output_path).duration
    except Exception as e:
        print(f"TTS request failed: {e}")
        return False, 0


from PIL import Image
import numpy as np

def add_logo(clip, logo_path=LOGO_PATH, opacity=0.75, duration=None, size_factor=1.5, padding=20):
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
    start_time = time.time()
    try:
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
        final_clip = add_logo(final_clip)

        # Optimize audio processing
        final_audio = CompositeAudioClip(audio_clips)
        final_audio = final_audio.audio_fadeout(1)  # 1 second fade-out at the end

        if bgm_path and os.path.exists(bgm_path):
            background_audio = AudioFileClip(bgm_path)
            background_audio = afx.audio_loop(background_audio, duration=final_clip.duration)
            background_audio = background_audio.audio_fadeout(1)
            final_audio = CompositeAudioClip([final_audio, background_audio.volumex(bgm_volume)])

        final_clip = final_clip.set_audio(final_audio)

        # Write the video directly
        final_clip.write_videofile(output_video_path, fps=24, codec='libx264', audio_codec='aac', audio_bitrate='128k', threads=4)
        logger.info(f"Video creation completed in {time.time() - start_time} seconds")

        return intro_duration, end_credits_duration

    except Exception as e:
        logging.error(f"Error in create_video: {str(e)}")
        if os.path.exists(output_video_path):
            os.remove(output_video_path)
        raise

    finally:
        if 'final_clip' in locals():
            final_clip.close()

import logging
import traceback
from moviepy.editor import VideoFileClip, ColorClip, CompositeVideoClip

logging.basicConfig(filename='/home/varun_saagar/logs/video_conversion.log', level=logging.INFO)

def convert_vertical_to_horizontal(input_path, output_path):
    logging.info(f"Starting conversion from vertical to horizontal: {input_path} -> {output_path}")
    clip = None
    final_clip = None
    try:
        # Load the vertical video
        clip = VideoFileClip(input_path)
        logging.info(f"Loaded vertical video. Duration: {clip.duration}, Size: {clip.size}")
        
        # Define the target resolution (e.g., 1920x1080 for Full HD)
        target_width = 1920
        target_height = 1080
        
        # Calculate the scaling factor to fit the height
        scale_factor = target_height / clip.h
        logging.info(f"Calculated scale factor: {scale_factor}")
        
        # Resize the clip while maintaining aspect ratio
        resized_clip = clip.resize(height=target_height)
        logging.info(f"Resized clip. New size: {resized_clip.size}")
        
        # Create a black background
        background = ColorClip(size=(target_width, target_height), color=(0,0,0))
        background = background.set_duration(clip.duration)
        logging.info("Created black background")
        
        # Calculate the position to center the video
        x_center = (target_width - resized_clip.w) / 2
        logging.info(f"Calculated x_center: {x_center}")
        
        # Composite the resized clip onto the background
        final_clip = CompositeVideoClip([background, resized_clip.set_position((x_center, 0))])
        logging.info("Created composite video clip")
        
        # Write the final horizontal video
        logging.info("Starting to write horizontal video")
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
        logging.info("Finished writing horizontal video")
        
        logging.info("Horizontal conversion completed successfully")
        logging.info(f"Horizontal video created: {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error creating horizontal video: {str(e)}")
        logging.error(traceback.format_exc())
        return False
    finally:
        # Close the clips
        if clip:
            clip.close()
        if final_clip:
            final_clip.close()
        logging.info("Closed all clips")

def process_slide(slide, i, image_folder, audio_folder, silence_duration, zoom_factor, zoom_duration):
    image_path = os.path.join(image_folder, slide['image']['local_path'])
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        overall_styles = slide['overall_styles']
        
        if i == 0:  # First slide
            webstories_title_data = (slide['image']['webstorie_title'], slide['title_styles'])
            title_data = (slide['image']['title'], slide['title_styles'])
            description_data = (slide['image']['description'], slide['description_styles'])
            tts_text = f"{slide['image']['webstorie_title']}. {slide['image']['title']}. {slide['image']['description']}"
        else:
            webstories_title_data = None
            title_data = (slide['image']['title'], slide['title_styles'])
            description_data = (slide['image']['description'], slide['description_styles'])
            tts_text = f"{slide['image']['title']}. {slide['image']['description']}"
        
        credits_data = (slide['image']['credit'], slide['credit_styles'])
        
        # Generate TTS for this slide
        tts_output_path = os.path.join(audio_folder, f"slide_{i+1}.wav")
        tts_success, tts_duration = generate_tts(tts_text, tts_output_path)
        
        if tts_success:
            slide_duration = max(tts_duration + silence_duration, zoom_duration)
        else:
            slide_duration = zoom_duration
        
        clip = process_image(img, overall_styles, title_data, description_data, credits_data, 
                            duration=slide_duration, zoom_factor=zoom_factor, zoom_duration=zoom_duration,
                            is_first_slide=(i==0), webstories_title_data=webstories_title_data)
        
        return clip, slide_duration
    else:
        print(f"Warning: Image not found: {image_path}")
        return None, 0


import shutil

def generate_tts_sequential(tts_data):
    tts_durations = []
    for i, tts_text, audio_folder in tts_data:
        tts_output_path = os.path.join(audio_folder, f"slide_{i+1}.wav")
        success, audio_duration = generate_tts(tts_text, tts_output_path)
        
        if success:
            print(f"TTS generated for slide {i+1} (duration: {audio_duration:.2f}s)")
        else:
            print(f"Failed to generate TTS for slide {i+1}")
            audio_duration = 0
        
        tts_durations.append(audio_duration)
    return tts_durations
import traceback

def main(url, bgm_path, zoom_factor, zoom_duration, bgm_volume, silence_duration, use_tts=True, output_format='vertical', intro_path=None, end_credits_path=None):
    unique_id = None
    main_folder = None
    vertical_gcp_path = None
    horizontal_gcp_path = None
    vertical_signed_url = None
    horizontal_signed_url = None

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

            tts_durations = generate_tts_sequential(tts_data)
        else:
            tts_durations = [0] * len(story_data)

        print("Step 4: Processing slides...")
        overlay_clips = []
        for i, slide in enumerate(story_data):
            clip, slide_duration = process_slide(slide, i, image_folder, audio_folder, tts_durations[i], silence_duration, zoom_factor, zoom_duration)
            if clip:
                overlay_clips.append(clip)

        total_audio_duration = sum(tts_durations) + (silence_duration * len(story_data))

        print("Step 5: Creating vertical video...")
        output_path = os.path.join(vertical_folder, f'{content_id}_vertical.mp4')
        intro_duration, end_credits_duration = create_video(
            image_folder, output_path, total_audio_duration, overlay_clips=overlay_clips, 
            bgm_path=bgm_path, bgm_volume=bgm_volume, 
            audio_folder=audio_folder if use_tts else None,
            silence_duration=silence_duration, output_format=output_format,
            intro_path=intro_path, end_credits_path=end_credits_path
        )

        print("Step 6: Converting vertical to horizontal...")
        if output_path and os.path.exists(output_path):
            horizontal_output_path = os.path.join(horizontal_folder, f'{content_id}_horizontal.mp4')
            conversion_success = convert_vertical_to_horizontal(output_path, horizontal_output_path)
            if conversion_success:
                print(f"Horizontal video created: {horizontal_output_path}")
            else:
                print("Failed to create horizontal video")
                log_error("Failed to create horizontal video")
        else:
            print("Vertical video not found, skipping horizontal conversion.")

        print("Step 7: Copying files to the 'contentgrowth' bucket...")
        if main_folder:
            try:
                copy_to_bucket(main_folder, 'contentgrowth')
                print(f"All files from {main_folder} have been copied to the 'contentgrowth/English' bucket.")
                
                vertical_gcp_path = f"English/{unique_id}/vertical/{content_id}_vertical.mp4"
                horizontal_gcp_path = f"English/{unique_id}/horizontal/{content_id}_horizontal.mp4"

                vertical_signed_url = generate_signed_url('contentgrowth', vertical_gcp_path)
                horizontal_signed_url = generate_signed_url('contentgrowth', horizontal_gcp_path)

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

        return vertical_gcp_path, horizontal_gcp_path, vertical_signed_url, horizontal_signed_url

    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        if unique_id:
            log_error(f"Article {unique_id}: {e}", traceback.format_exc())
        raise


def log_error(message, details=None):
    log_file = ERROR_LOG_FILE
    with open(log_file, "a") as f:
        f.write(f"\n{message}\n")
        if details:
            f.write(details)

@celery.task(bind=True)
def create_video_task(self, url, bgm_path, zoom_factor, zoom_duration, bgm_volume, silence_duration, use_tts, output_format, intro_path, end_credits_path):
    try:
        vertical_gcp_path, horizontal_gcp_path, vertical_signed_url, horizontal_signed_url = main(
            url, bgm_path, zoom_factor, zoom_duration, bgm_volume, silence_duration, use_tts,
            output_format, intro_path, end_credits_path
        )
        
        return {
            "vertical_video_url": vertical_signed_url,
            "horizontal_video_url": horizontal_signed_url,
            "vertical_gcp_path": vertical_gcp_path,
            "horizontal_gcp_path": horizontal_gcp_path,
            "status": "Completed"
        }
    except Exception as e:
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

print("Defining routes in api_videocreation.py")

def create_video_api():
    print("create_video_api function called")
    try:
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

        task = create_video_task.delay(
            url, bgm_path, zoom_factor, zoom_duration, bgm_volume, silence_duration, use_tts,
            'vertical', intro_path, end_credits_path
        )
        
        return jsonify({"task_id": task.id, "status": "Processing"}), 202
    except Exception as e:
        logger.error(f"Error in create_video_api: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500


def get_status(task_id):
    task = create_video_task.AsyncResult(task_id)
    try:
        task = create_video_task.AsyncResult(task_id)
        if task.state == 'PENDING':
            response = {
                'state': task.state,
                'status': 'Pending...'
            }
        elif task.state != 'FAILURE':
            response = {
                'state': task.state,
                'status': task.info.get('status', '')
            }
            if 'vertical_video_url' in task.info:
                response['vertical_video_url'] = task.info['vertical_video_url']
                response['horizontal_video_url'] = task.info['horizontal_video_url']
                response['vertical_gcp_path'] = task.info['vertical_gcp_path']
                response['horizontal_gcp_path'] = task.info['horizontal_gcp_path']
        else:
            response = {
                'state': task.state,
                'status': 'Failed',
                'error': str(task.info)
            }
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in get_status: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500

def health_check():
    try:
        # Check if Redis is accessible
        redis_client = redis.Redis.from_url(app.config['CELERY_BROKER_URL'])
        redis_client.ping()

        # Check if Celery worker is running
        i = celery.control.inspect()
        if not i.ping():
            raise Exception("No running Celery workers found")

        return jsonify({"status": "healthy", "message": "API and dependencies are functioning correctly"}), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "unhealthy", "message": str(e)}), 500

def create_app():
    init_routes(app)
    return app

def init_routes(app):
    print("Initializing routes...")
    if 'create_video_api' not in app.view_functions:
        app.add_url_rule('/create_video', 'create_video_api', create_video_api, methods=['POST'])
    
    if 'get_status' not in app.view_functions:
        app.add_url_rule('/status/<task_id>', 'get_status', get_status, methods=['GET'])
    
    if 'health_check' not in app.view_functions:
        app.add_url_rule('/health', 'health_check', health_check, methods=['GET'])

def create_app():
    init_routes(app)
    return app

if __name__ == '__main__':
    print(f"Executing api_videocreation.py. __name__ is: {__name__}")
    print(f"sys.modules keys: {list(sys.modules.keys())}")
    create_app()
    app.run(host='0.0.0.0', port=3000, debug=False)
else:
    print(f"Executing api_videocreation.py. __name__ is: {__name__}")
    print("api_videocreation.py is being imported, not initializing routes.")