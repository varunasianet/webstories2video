import os
import logging
import time
import requests
import json
from datetime import datetime, timedelta
import pytz
from sqlalchemy import create_engine, text
from google.cloud.sql.connector import Connector
from google.cloud.storage import Client as storage_client
from google.oauth2 import service_account
import threading
import queue
from tasks import create_video_task

# Define BASE_DIR
BASE_DIR = "/home/varun_saagar/videocreation"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define IST timezone
ist = pytz.timezone('Asia/Kolkata')

# API endpoints
VIDEO_CREATION_API_URL = "http://localhost:3000/create_video"
VIDEO_STATUS_API_URL = "http://localhost:3000/status/"
VIDEO_POST_API_URL = "https://core-webservices-api.asianetnews.com/ai/video/add"

# Database connection details
INSTANCE_CONNECTION_NAME = "asianet-tech-staging:asia-south1:webstories-asianet-db"
DB_USER = "webstories-user"
DB_PASS = "asianetweb"
DB_NAME = "webstoriesrss"

# Google Cloud Storage details
BUCKET_NAME = "contentgrowth"
SERVICE_ACCOUNT_FILE = "/home/varun_saagar/videocreation/creds/asianet-tech-staging-91b6f4c817e0.json"

# Semaphore to limit concurrent video creations
video_semaphore = threading.Semaphore(3)

# Queue for pending webstories
webstory_queue = queue.Queue()

def init_connection_engine():
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

    engine = create_engine(
        "mysql+pymysql://",
        creator=getconn,
    )
    return engine


def get_pending_webstories(engine):
    with engine.connect() as connection:
        query = text("""
            SELECT webstory_id, link
            FROM webstories
            WHERE video_available_status = 'NO'
            ORDER BY sync_time DESC
            LIMIT 10
        """)
        result = connection.execute(query)
        return result.fetchall()

def update_webstory_status(engine, webstory_id, vertical_gcp_path, horizontal_gcp_path, vertical_video_url, horizontal_video_url, status):
    try:
        with engine.connect() as connection:
            query = text("""
                UPDATE webstories
                SET video_creation_endtime = :end_time,
                    vertical_video_path = :vertical_path,
                    horizontal_video_path = :horizontal_path,
                    vertical_video_url = :vertical_video_url,
                    horizontal_video_url = :horizontal_video_url,
                    video_available_status = :status
                WHERE webstory_id = :webstory_id
            """)
            connection.execute(query, {
                "end_time": datetime.now(ist),
                "vertical_path": vertical_gcp_path,
                "horizontal_path": horizontal_gcp_path,
                "vertical_video_url": vertical_video_url,
                "horizontal_video_url": horizontal_video_url,
                "status": status,
                "webstory_id": webstory_id
            })
            connection.commit()
    except Exception as e:
        logger.error(f"Database error when updating webstory status: {str(e)}")




def create_video(url, zoom_factor, zoom_duration, bgm_volume, silence_duration, 
                 use_tts, output_format, intro_path, end_credits_path):
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "url": url,
        "zoom_factor": zoom_factor,
        "zoom_duration": zoom_duration,
        "bgm_volume": bgm_volume,
        "silence_duration": silence_duration,
        "use_tts": use_tts,
        "output_format": output_format,
        "intro_path": intro_path,
        "end_credits_path": end_credits_path
    }
    try:
        response = requests.post(VIDEO_CREATION_API_URL, data=json.dumps(data), headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error occurred: {e}")
        logger.error(f"Response content: {e.response.content}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred while making the request: {e}")
        return None


def check_video_status(task_id):
    try:
        response = requests.get(f"{VIDEO_STATUS_API_URL}{task_id}")
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to get status for task ID: {task_id}. Status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred while checking video status: {e}")
        return None

def update_posted_to_api_status(engine, webstory_id, status):
    with engine.connect() as connection:
        query = text("""
            UPDATE webstories
            SET posted_to_api = :status
            WHERE webstory_id = :webstory_id
        """)
        connection.execute(query, {
            "status": status,
            "webstory_id": webstory_id
        })
        connection.commit()


def post_video_urls(engine, webstory_id, article_url, vertical_video_url, horizontal_video_url):
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "articleUrl": article_url,
        "vertical_video": vertical_video_url,
        "horizontal_video": horizontal_video_url
    }
    try:
        response = requests.post(VIDEO_POST_API_URL, json=data, headers=headers)
        response.raise_for_status()  # This will raise an HTTPError for bad responses
        logger.info(f"Successfully posted video URLs for article: {article_url}")
        update_posted_to_api_status(engine, webstory_id, 'YES')
        return True
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error occurred while posting video URLs for article: {article_url}. Status code: {e.response.status_code}")
        logger.error(f"Response content: {e.response.text}")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred while posting video URLs for article: {article_url}. Error: {str(e)}")
        return False



def update_video_creation_start_time(engine, webstory_id):
    with engine.connect() as connection:
        query = text("""
            UPDATE webstories
            SET video_creation_starttime = :start_time
            WHERE webstory_id = :webstory_id
        """)
        connection.execute(query, {
            "start_time": datetime.now(ist),
            "webstory_id": webstory_id
        })
        connection.commit()



def process_single_webstory(engine, webstory_id, link, zoom_factor, zoom_duration, 
                            bgm_volume, silence_duration, use_tts, output_format, 
                            intro_path, end_credits_path):
    logger.info(f"Creating video for webstory ID: {webstory_id}")
    update_video_creation_start_time(engine, webstory_id)
    result = create_video(link, zoom_factor, zoom_duration, bgm_volume, silence_duration, 
                          use_tts, output_format, intro_path, end_credits_path)
    if result and 'task_id' in result:
        task_id = result['task_id']
        max_attempts = 60  # 30 minutes (60 * 30 seconds)
        attempts = 0
        while attempts < max_attempts:
            status = check_video_status(task_id)
            if status['state'] == 'SUCCESS':
                vertical_url = status.get('vertical_video_url', '')
                horizontal_url = status.get('horizontal_video_url', '')
                
                update_webstory_status(engine, webstory_id, vertical_url, horizontal_url, vertical_url, horizontal_url, 'YES')
                
                # Post video URLs as soon as they are available
                post_video_urls(engine, webstory_id, link, vertical_url, horizontal_url)
                
                logger.info(f"Video created and status updated for webstory ID: {webstory_id}")
                return
            elif status['state'] in ['FAILURE', 'REVOKED']:
                logger.error(f"Video creation failed for webstory ID: {webstory_id}")
                update_webstory_status(engine, webstory_id, '', '', '', '', 'FAILED')
                return
            else:
                time.sleep(30)  # Wait for 30 seconds
                attempts += 1
        
        logger.error(f"Timeout: Video creation process took too long for webstory ID: {webstory_id}")
        update_webstory_status(engine, webstory_id, '', '', '', '', 'TIMEOUT')
    else:
        logger.error(f"Failed to initiate video creation for webstory ID: {webstory_id}")
        update_webstory_status(engine, webstory_id, '', '', '', '', 'FAILED')

from celery.result import AsyncResult
import celery.exceptions
import time
import logging
import os

logger = logging.getLogger(__name__)

from celery.result import AsyncResult
import celery.exceptions
import time
import logging
import os

logger = logging.getLogger(__name__)

def handle_video_creation(engine, webstory_id, url, task):
    try:
        logger.info(f"Waiting for task result for webstory ID: {webstory_id}")
        result = AsyncResult(task.id).get(timeout=7500)  # 125 minutes timeout
        logger.info(f"Task result received for webstory ID: {webstory_id}: {result}")
        
        if isinstance(result, dict) and 'vertical_video_url' in result and 'horizontal_video_url' in result:
            vertical_gcp_path = result['vertical_gcp_path']
            horizontal_gcp_path = result['horizontal_gcp_path']
            vertical_signed_url = result['vertical_video_url']
            horizontal_signed_url = result['horizontal_video_url']
            
            logger.info(f"Updating webstory status for ID: {webstory_id}")
            update_webstory_status(engine, webstory_id, vertical_gcp_path, horizontal_gcp_path, vertical_signed_url, horizontal_signed_url, 'YES')
            
            logger.info(f"Posting video URLs for webstory ID: {webstory_id}")
            post_success = post_video_urls(engine, webstory_id, url, vertical_signed_url, horizontal_signed_url)
            
            if post_success:
                logger.info(f"Video URLs posted successfully for webstory ID: {webstory_id}")
            else:
                logger.error(f"Failed to post video URLs for webstory ID: {webstory_id}")
            
            return result
        else:
            logger.error(f"Unexpected result format for URL {url}: {result}")
            update_webstory_status(engine, webstory_id, '', '', '', '', 'FAILED')
            return None
    except celery.exceptions.TimeoutError:
        logger.error(f"Task timed out for URL {url}")
        update_webstory_status(engine, webstory_id, '', '', '', '', 'TIMEOUT')
        return None
    except Exception as e:
        logger.error(f"Error processing task result for URL {url}: {str(e)}")
        logger.error(traceback.format_exc())
        update_webstory_status(engine, webstory_id, '', '', '', '', 'FAILED')
        return None

def process_pending_webstories(engine):
    while True:
        try:
            logger.info("Fetching pending webstories...")
            pending_webstories = get_pending_webstories(engine)
            if pending_webstories:
                logger.info(f"Found {len(pending_webstories)} pending webstories.")
                for webstory_id, link in pending_webstories:
                    logger.info(f"Submitting task for webstory ID: {webstory_id}, URL: {link}")
                    task = create_video_task.delay(
                        link,
                        os.path.join(BASE_DIR, "Background_music", "happy", "07augsug.mp3"),
                        1.1,
                        5,
                        0.1,
                        0.5,
                        True,
                        'mp4',
                        os.path.join(BASE_DIR, "artificts", "intro-fixed.mp4"),
                        os.path.join(BASE_DIR, "artificts", "Newsable end sting__Vertical.mp4")
                    )
                    logger.info(f"Task submitted for webstory ID: {webstory_id}, Task ID: {task.id}")
                    
                    result = handle_video_creation(engine, webstory_id, link, task)
                    if result:
                        logger.info(f"Successfully processed webstory ID: {webstory_id}")
                    else:
                        logger.error(f"Failed to process webstory ID: {webstory_id}")

                logger.info("Finished processing batch of webstories. Waiting for 60 seconds before next check...")
                time.sleep(60)
            else:
                logger.info("No pending webstories. Waiting for 60 seconds before next check...")
                time.sleep(60)
        except Exception as e:
            logger.error(f"An error occurred while processing webstories: {str(e)}")
            time.sleep(60)  # Wait for a minute before retrying

def generate_signed_url(bucket_name, blob_name):
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
    client = storage_client(credentials=credentials)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.utcnow() + timedelta(hours=1),
        method="GET",
    )
    return url

def main():
    try:
        engine = init_connection_engine()
        process_pending_webstories(engine)
    except KeyboardInterrupt:
        logger.info("Process interrupted. Shutting down gracefully...")
    except Exception as e:
        logger.error(f"An unexpected error occurred in main: {str(e)}")


if __name__ == "__main__":
    engine = init_connection_engine()
    process_pending_webstories(engine)


