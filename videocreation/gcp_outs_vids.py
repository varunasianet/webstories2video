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
from moviepy.audio.fx.all import audio_fadeout
from requests.exceptions import RequestException, Timeout, ConnectionError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry 
from io import BytesIO
from google.cloud import storage
from google.auth import default

# Set up GCS client using default credentials
credentials, project = default()
storage_client = storage.Client(credentials=credentials, project=project)
bucket_name = 'contentgrowth'
bucket = storage_client.bucket(bucket_name)

#helpers
def upload_file_to_gcs(local_path, gcs_path):
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"File uploaded to GCS: gs://{bucket_name}/{gcs_path}")

def download_file_from_gcs(gcs_path, local_path):
    blob = bucket.blob(gcs_path)
    blob.download_to_filename(local_path)
    print(f"File downloaded from GCS: gs://{bucket_name}/{gcs_path}")

def check_file_exists_in_gcs(gcs_path):
    blob = bucket.blob(gcs_path)
    return blob.exists()

def list_files_in_gcs_folder(folder_path):
    blobs = bucket.list_blobs(prefix=folder_path)
    return [blob.name for blob in blobs]

def is_gcs_path(path):
    return path.startswith('gs://') or not path.startswith('/')


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
    # Check if img is a NumPy array (from OpenCV)
    if isinstance(img, np.ndarray):
        # Convert OpenCV image (BGR) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Convert NumPy array to PIL Image
        pil_img = Image.fromarray(img)
    else:
        # If it's already a PIL Image, use it directly
        pil_img = img

    # Resize the image using PIL
    pil_img = pil_img.resize((DEFAULT_WIDTH, DEFAULT_HEIGHT), Image.LANCZOS)
    
    # Convert back to NumPy array for MoviePy
    img = np.array(pil_img)
    
    bg_clip = ImageClip(img).set_duration(duration)

    # Apply smooth zoom effect
    bg_clip = zoom(bg_clip, zoom_factor, zoom_duration)

    if is_first_slide and webstories_title_data:
        overlay_clip = create_overlay((DEFAULT_WIDTH, DEFAULT_HEIGHT), overall_styles, title_data, description_data, credits_data, webstories_title_data, is_first_slide=True).set_duration(duration)
    else:
        overlay_clip = create_overlay((DEFAULT_WIDTH, DEFAULT_HEIGHT), overall_styles, title_data, description_data, credits_data, is_first_slide=is_first_slide).set_duration(duration)

    final_clip = CompositeVideoClip([bg_clip, overlay_clip], size=(DEFAULT_WIDTH, DEFAULT_HEIGHT)).set_duration(duration)

    return final_clip

def save_to_gcs(local_path, gcs_path):
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"File uploaded to GCS: gs://{bucket_name}/{gcs_path}")

def save_data(unique_id, story_data):
    main_folder = f'scraped_data/{unique_id}'
    vertical_folder = f'{main_folder}/vertical'
    horizontal_folder = f'{main_folder}/horizontal'
    image_folder = f'{main_folder}/Image'

    # Save JSON data
    json_path = f'{main_folder}/data.json'
    blob = bucket.blob(json_path)
    blob.upload_from_string(json.dumps(story_data, ensure_ascii=False, indent=2))

    # Upload images to GCS
    for slide in story_data:
        img_url = slide['image'].get('url')
        if img_url:
            response = requests.get(img_url)
            if response.status_code == 200:
                image_filename = os.path.basename(slide['image']['local_path'])
                gcs_image_path = f"{image_folder}/{image_filename}"
                blob = bucket.blob(gcs_image_path)
                blob.upload_from_string(response.content, content_type='image/jpeg')
                slide['image']['gcs_path'] = gcs_image_path

    print(f"Data saved for article {unique_id}")
    print(f"Vertical folder path: gs://{bucket_name}/{vertical_folder}")
    print(f"Horizontal folder path: gs://{bucket_name}/{horizontal_folder}")

def download_and_preprocess_image(url, folder_path, filename):
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        img = crop_and_resize_image(img, (DEFAULT_WIDTH, DEFAULT_HEIGHT))
        
        # Save to a temporary local file
        temp_path = f"/tmp/{filename}"
        img.save(temp_path, 'JPEG')
        
        # Upload to GCS
        gcs_path = f"{folder_path}/{filename}"
        save_to_gcs(temp_path, gcs_path)
        
        # Remove temporary local file
        os.remove(temp_path)
        
        return gcs_path
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
        response = session.post(TTS_URL, data=data, timeout=300)  # Increased timeout to 5 minutes
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(response.content)
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

    # Process overlay clips
    if overlay_clips:
        for i, clip in enumerate(overlay_clips):
            clips.append(clip)
            
            if audio_folder:
                audio_path = f"{audio_folder}/slide_{i+1}.wav"
                if check_file_exists_in_gcs(audio_path):
                    temp_audio_path = f"/tmp/slide_{i+1}.wav"
                    download_file_from_gcs(audio_path, temp_audio_path)
                    
                    # Load the audio file using pydub
                    slide_audio = AudioSegment.from_wav(temp_audio_path)
                    
                    # Add silence at the beginning
                    silence = AudioSegment.silent(duration=silence_duration * 1000)
                    full_audio = silence + slide_audio
                    
                    # Apply fade out
                    fade_duration = min(500, len(full_audio))  # 500ms fade out, or shorter if the audio is very short
                    full_audio = full_audio.fade_out(duration=fade_duration)
                    
                    # Export the processed audio
                    processed_audio_file = f"/tmp/processed_slide_{i+1}.wav"
                    full_audio.export(processed_audio_file, format="wav")
                    
                    # Use the processed audio in the video clip
                    clip_audio = AudioFileClip(processed_audio_file)
                    clip = clip.set_audio(clip_audio)
                    clip_duration = len(full_audio) / 1000.0  # Duration in seconds
                    clip = clip.set_duration(clip_duration)
                    audio_clips.append(clip_audio)
                    
                    os.remove(temp_audio_path)
                    os.remove(processed_audio_file)
                else:
                    print(f"Warning: Audio file not found in GCS: {audio_path}")
                    clip = clip.set_duration(silence_duration)
                    audio_clips.append(AudioClip(lambda t: 0, duration=silence_duration))
            else:
                clip = clip.set_duration(silence_duration)
                audio_clips.append(AudioClip(lambda t: 0, duration=silence_duration))

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

    # Write the final video to a temporary local file
    temp_output_path = f"/tmp/{os.path.basename(output_video_path)}"
    final_clip.write_videofile(temp_output_path, fps=24, codec='libx264', audio_codec='aac', audio_bitrate='128k',
                               preset='faster', threads=cpu_count())

    # Upload the video to GCS
    upload_file_to_gcs(temp_output_path, output_video_path)

    # Remove the temporary local file
    os.remove(temp_output_path)

    return intro_duration, end_credits_duration

def convert_vertical_to_horizontal(input_path, output_path):
    # Download the vertical video from GCS
    temp_input_path = f"/tmp/{os.path.basename(input_path)}"
    download_file_from_gcs(input_path, temp_input_path)

    # Load the vertical video
    clip = VideoFileClip(temp_input_path)
    
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
    
    # Write the final horizontal video to a temporary local file
    temp_output_path = f"/tmp/{os.path.basename(output_path)}"
    final_clip.write_videofile(temp_output_path, codec='libx264', audio_codec='aac')
    
    # Upload the horizontal video to GCS
    upload_file_to_gcs(temp_output_path, output_path)
    
    # Close the clips and remove temporary files
    clip.close()
    final_clip.close()
    os.remove(temp_input_path)
    os.remove(temp_output_path)


def process_slide(slide, i, image_folder, audio_folder, use_tts, silence_duration, zoom_factor, zoom_duration):
    image_path = slide['image']['gcs_path']
    if check_file_exists_in_gcs(image_path):
        # Download the image to a temporary file
        temp_image_path = f"/tmp/{os.path.basename(image_path)}"
        download_file_from_gcs(image_path, temp_image_path)
        
        img = cv2.imread(temp_image_path)
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
        
        # Generate TTS for the slide description only if use_tts is True
        if use_tts:
            tts_output_path = f"/tmp/slide_{i+1}.wav"
            success, audio_duration = generate_tts(tts_text, tts_output_path)
            
            if success:
                print(f"TTS generated for slide {i+1} (duration: {audio_duration:.2f}s)")
                # Upload the audio file to GCS
                gcs_audio_path = f"{audio_folder}/slide_{i+1}.wav"
                upload_file_to_gcs(tts_output_path, gcs_audio_path)
                os.remove(tts_output_path)
            else:
                print(f"Failed to generate TTS for slide {i+1}")
                audio_duration = 0
        else:
            audio_duration = 0
        
        slide_duration = max(audio_duration + silence_duration, zoom_duration)
        clip = process_image(img, overall_styles, title_data, description_data, credits_data, 
                            duration=slide_duration, zoom_factor=zoom_factor, zoom_duration=zoom_duration,
                            is_first_slide=(i==0), webstories_title_data=webstories_title_data)
        
        os.remove(temp_image_path)
        return clip, slide_duration, tts_text
    else:
        print(f"Warning: Image not found in GCS: {image_path}")
        return None, 0, ""

def main(url, bgm_path, zoom_factor, zoom_duration, bgm_volume, silence_duration, use_tts=True, output_format='vertical', intro_path=None, end_credits_path=None):
    try:
        result = process_amp_story_pages(url)
        if not result:
            raise ValueError("Failed to scrape data from the URL")

        unique_id, story_data = result
        content_id = unique_id

        for slide in story_data:
            slide['content_id'] = content_id

        save_data(unique_id, story_data)
        print("All data has been scraped and saved.")

        base_dir = 'scraped_data'
        main_folder = f'{base_dir}/{unique_id}'
        vertical_folder = f'{main_folder}/vertical'
        horizontal_folder = f'{main_folder}/horizontal'

        image_folder = f'{main_folder}/Image'
        audio_folder = f'{main_folder}/Audio'

        overlay_clips = []
        slide_durations = []
        tts_texts = []

        for i, slide in enumerate(story_data):
            clip, slide_duration, tts_text = process_slide(slide, i, image_folder, audio_folder, use_tts, silence_duration, zoom_factor, zoom_duration)
            if clip:
                overlay_clips.append(clip)
                slide_durations.append(slide_duration)
                tts_texts.append(tts_text)

        total_duration = sum(slide_durations)

        output_path = f'{vertical_folder}/{content_id}_vertical.mp4'
        intro_duration, end_credits_duration = create_video(
            image_folder, output_path, total_duration, overlay_clips=overlay_clips, 
            bgm_path=bgm_path, bgm_volume=bgm_volume, 
            audio_folder=audio_folder if use_tts else None,
            silence_duration=silence_duration, output_format=output_format,
            intro_path=intro_path, end_credits_path=end_credits_path
        )

        print(f"Vertical video created: gs://{bucket_name}/{output_path}")
        print(f"Intro duration: {intro_duration:.2f} seconds")
        print(f"End credits duration: {end_credits_duration:.2f} seconds")
        print(f"Total video duration: {total_duration + intro_duration + end_credits_duration:.2f} seconds")

        # Print TTS texts for verification
        for i, text in enumerate(tts_texts):
            print(f"Slide {i+1} TTS text: {text}")

        horizontal_output_path = f'{horizontal_folder}/{content_id}_horizontal.mp4'
        convert_vertical_to_horizontal(output_path, horizontal_output_path)
        print(f"Horizontal video created: gs://{bucket_name}/{horizontal_output_path}")

    except ValueError as ve:
        print(f"Validation error: {ve}")
        if 'unique_id' in locals():
            log_error(f"Article {unique_id}: {ve}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        if 'unique_id' in locals():
            log_error(f"Article {unique_id}: {e}", traceback.format_exc())

def log_error(message, details=None):
    log_file = 'error_log.txt'
    blob = bucket.blob(log_file)
    
    # Download existing content
    existing_content = ''
    if blob.exists():
        existing_content = blob.download_as_text()
    
    # Append new error message
    new_content = f"{existing_content}\n{message}\n"
    if details:
        new_content += f"{details}\n"
    
    # Upload updated content
    blob.upload_from_string(new_content)

if __name__ == "__main__":
    url = "https://newsable.asianetnews.com/webstories/business/petrol-diesel-fresh-prices-announced-on-june-24-check-city-wise-rate-ajr-sfkcne"
    bgm_path = f"{BASE_DIR}/Background_music/happy/07augsug.mp3"
    zoom_factor = 1.15
    zoom_duration = 20
    bgm_volume = 0.15
    silence_duration = 1
    use_tts = True
    intro_path = f"{BASE_DIR}/artificts/intro-fixed.mp4"
    end_credits_path = f"{BASE_DIR}/artificts/Newsable end sting__Vertical.mp4"


    try:
        print("Starting video creation process...")
        main(url, bgm_path, zoom_factor, zoom_duration, bgm_volume, silence_duration, use_tts, 
             output_format='vertical', intro_path=intro_path, end_credits_path=end_credits_path)
        print("Video creation process completed successfully.")
    except Exception as e:
        print(f"An error occurred during the video creation process: {str(e)}")
        log_error("Main script execution error", str(e))

    # Additional error handling and cleanup
    try:
        # Clean up temporary files
        temp_dir = "/tmp/video_creation_temp"
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
            print(f"Temporary directory {temp_dir} has been cleaned up.")
    except Exception as cleanup_error:
        print(f"An error occurred during cleanup: {str(cleanup_error)}")
        log_error("Cleanup error", str(cleanup_error))

    import time
    start_time = time.time()

    try:
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\nTotal execution time: {total_time:.2f} seconds")
    except NameError:
        print("\nNote: Execution time measurement was not available for this run.")

    print("\nScript execution completed.")