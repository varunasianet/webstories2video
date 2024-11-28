import requests
import json
import time
import sqlalchemy
from google.cloud.sql.connector import Connector
import logging
from datetime import datetime, timedelta
import pytz
from google.cloud import storage
from google.oauth2 import service_account
import concurrent.futures
import threading
import queue
from sqlalchemy.exc import SQLAlchemyError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define IST timezone
ist = pytz.timezone('Asia/Kolkata')

# API endpoints
VIDEO_CREATION_API_URL = "http://localhost:5050/create_video"  # Replace with your actual API URL
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
# Replace the existing video_semaphore initialization with this:

# Configuration
MAX_CONCURRENT_VIDEOS = 3  # Default value, can be adjusted as needed
video_semaphore = threading.Semaphore(MAX_CONCURRENT_VIDEOS)

def update_max_concurrent_videos(new_value):
    global MAX_CONCURRENT_VIDEOS
    global video_semaphore
    
    MAX_CONCURRENT_VIDEOS = new_value
    video_semaphore = threading.Semaphore(MAX_CONCURRENT_VIDEOS)
    logger.info(f"Updated MAX_CONCURRENT_VIDEOS to {MAX_CONCURRENT_VIDEOS}")

# Queue for pending webstories
webstory_queue = queue.Queue()

from sqlalchemy.pool import QueuePool

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

    engine = sqlalchemy.create_engine(
        "mysql+pymysql://",
        creator=getconn,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10
    )
    return engine


def execute_with_retry(connection, query, params=None, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = connection.execute(query, params)
            connection.commit()
            return result
        except SQLAlchemyError as e:
            logger.error(f"Database error (attempt {attempt + 1}): {str(e)}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff

def get_pending_webstories(engine):
    with engine.connect() as connection:
        query = sqlalchemy.text("""
            SELECT webstory_id, link, posted_to_api, video_available_status, processing_status
            FROM webstories
            WHERE (posted_to_api = 'NO' OR posted_to_api IS NULL)
               AND (processing_status IS NULL OR processing_status = 'FAILED')
               AND (video_available_status IS NULL OR video_available_status = 'NO')
            ORDER BY sync_time DESC
        """)
        result = connection.execute(query)
        return result.fetchall()


def get_video_urls(engine, webstory_id):
    with engine.connect() as connection:
        query = sqlalchemy.text("""
            SELECT vertical_video_url, horizontal_video_url
            FROM webstories
            WHERE webstory_id = :webstory_id
        """)
        result = connection.execute(query, {"webstory_id": webstory_id}).fetchone()
        return result[0], result[1] if result else (None, None)

def generate_signed_url(bucket_name, blob_name):
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.utcnow() + timedelta(hours=1),
        method="GET",
    )
    return url


def truncate_string(s, max_length):
    return s[:max_length] if s else s

def update_webstory_status(engine, webstory_id, vertical_path, horizontal_path, vertical_video_url, horizontal_video_url, status, processing_status):
    max_length = 255
    with engine.connect() as connection:
        query = sqlalchemy.text("""
            UPDATE webstories
            SET video_creation_starttime = :start_time,
                video_creation_endtime = :end_time,
                vertical_video_path = :vertical_path,
                horizontal_video_path = :horizontal_path,
                vertical_video_url = :vertical_video_url,
                horizontal_video_url = :horizontal_video_url,
                video_available_status = :status,
                processing_status = :processing_status
            WHERE webstory_id = :webstory_id
        """)
        params = {
            "start_time": datetime.now(ist),
            "end_time": datetime.now(ist),
            "vertical_path": truncate_string(vertical_path, max_length),
            "horizontal_path": truncate_string(horizontal_path, max_length),
            "vertical_video_url": truncate_string(vertical_video_url, max_length),
            "horizontal_video_url": truncate_string(horizontal_video_url, max_length),
            "status": status,
            "processing_status": processing_status,
            "webstory_id": webstory_id
        }
        execute_with_retry(connection, query, params)



def update_posted_to_api_status(engine, webstory_id, status, timestamp=None):
    with engine.connect() as connection:
        query = sqlalchemy.text("""
            UPDATE webstories
            SET posted_to_api = :status,
                api_response_timestamp = :timestamp
            WHERE webstory_id = :webstory_id
        """)
        params = {
            "status": status,
            "timestamp": timestamp,
            "webstory_id": webstory_id
        }
        execute_with_retry(connection, query, params)

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
        response = requests.post(VIDEO_POST_API_URL, json=data, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        response_data = response.json()
        
        if "articleUrl" in response_data:
            timestamp = response_data.get("timestamp")
            logger.info(f"Successfully posted video URLs for article: {article_url}")
            logger.info(f"Response timestamp: {timestamp}")
            
            update_posted_to_api_status(engine, webstory_id, 'YES', timestamp)
            update_webstory_status(engine, webstory_id, '', '', '', '', 'YES', 'COMPLETED')
            return True
        else:
            logger.warning(f"Unexpected response format for article: {article_url}. Response: {response_data}")
            update_posted_to_api_status(engine, webstory_id, 'NO')
            update_webstory_status(engine, webstory_id, '', '', '', '', 'FAILED', 'FAILED')
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred while posting video URLs: {e}")
        update_posted_to_api_status(engine, webstory_id, 'NO')
        update_webstory_status(engine, webstory_id, '', '', '', '', 'FAILED', 'FAILED')
        return False

# Add a timeout for requests
REQUEST_TIMEOUT = 3000 

def create_video(url, max_retries=3, retry_delay=30):
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "url": url
    }
    for attempt in range(max_retries):
        try:
            logger.info(f"Sending request to create video for URL: {url} (Attempt {attempt + 1})")
            response = requests.post(VIDEO_CREATION_API_URL, json=data, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            result = response.json()
            if "vertical_video_url" in result and "horizontal_video_url" in result:
                logger.info(f"Video creation successful for URL: {url}")
                return result
            else:
                logger.error(f"Unexpected response format for video creation: {result}")
                return None
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout occurred while creating video for URL: {url} (Attempt {attempt + 1})")
        except requests.exceptions.RequestException as e:
            logger.error(f"An error occurred while creating video for URL: {url}: {e}")
        
        if attempt < max_retries - 1:
            logger.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            logger.error(f"Max retries reached for URL: {url}")
            return None

def poll_video_status(url, job_id, max_attempts=60, poll_interval=30):
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{VIDEO_CREATION_API_URL}/status/{job_id}", timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            result = response.json()
            if result.get("status") == "completed":
                logger.info(f"Video creation completed for URL: {url}")
                return result
            elif result.get("status") == "failed":
                logger.error(f"Video creation failed for URL: {url}")
                return None
            else:
                logger.info(f"Video creation still in progress for URL: {url}. Waiting...")
                time.sleep(poll_interval)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error while polling video status for URL: {url}: {e}")
            time.sleep(poll_interval)
    
    logger.error(f"Polling timeout reached for URL: {url}")
    return None

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
        response = requests.post(VIDEO_POST_API_URL, json=data, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        logger.info(f"Successfully posted video URLs for article: {article_url}")
        update_posted_to_api_status(engine, webstory_id, 'YES')
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred while posting video URLs: {e}")
        update_posted_to_api_status(engine, webstory_id, 'NO')
        return False


def process_single_webstory(engine, webstory_id, link):
    try:
        logger.info(f"Processing webstory ID: {webstory_id}")
        
        # Update status to PROCESSING
        update_webstory_status(engine, webstory_id, '', '', '', '', 'PROCESSING', 'PROCESSING')
        
        result = create_video(link)
        if result and "vertical_video_url" in result and "horizontal_video_url" in result:
            vertical_path = result["vertical_video_url"]
            horizontal_path = result["horizontal_video_url"]
            
            logger.info(f"Updating database for webstory ID: {webstory_id}")
            update_webstory_status(engine, webstory_id, vertical_path, horizontal_path, vertical_path, horizontal_path, 'YES', 'COMPLETED')
            
            logger.info(f"Posting video URLs for webstory ID: {webstory_id}")
            if post_video_urls(engine, webstory_id, link, vertical_path, horizontal_path):
                logger.info(f"Video created, status updated, and URLs posted for webstory ID: {webstory_id}")
            else:
                logger.warning(f"Video created but failed to post URLs for webstory ID: {webstory_id}")
                update_posted_to_api_status(engine, webstory_id, 'NO')
                update_webstory_status(engine, webstory_id, vertical_path, horizontal_path, vertical_path, horizontal_path, 'YES', 'FAILED')
        else:
            logger.warning(f"Failed to create video for webstory ID: {webstory_id}")
            update_webstory_status(engine, webstory_id, '', '', '', '', 'FAILED', 'FAILED')
            update_posted_to_api_status(engine, webstory_id, 'NO')
    except Exception as e:
        logger.error(f"Error processing webstory ID {webstory_id}: {str(e)}")
        update_webstory_status(engine, webstory_id, '', '', '', '', 'FAILED', 'FAILED')
        update_posted_to_api_status(engine, webstory_id, 'NO')



def process_pending_webstories(engine):
    processed_webstories = set()
    while True:
        try:
            webstory = webstory_queue.get(timeout=1)  # Wait for 1 second
            webstory_id, link, posted_to_api, video_available_status, processing_status = webstory
            if webstory_id not in processed_webstories:
                try:
                    process_single_webstory(engine, webstory_id, link)
                    processed_webstories.add(webstory_id)
                except Exception as e:
                    logger.error(f"Error processing webstory ID {webstory_id}: {str(e)}")
                    update_webstory_status(engine, webstory_id, '', '', '', '', 'FAILED', 'FAILED')
            webstory_queue.task_done()
        except queue.Empty:
            logger.info("Queue is empty. Checking for new webstories...")
            pending_webstories = get_pending_webstories(engine)
            new_webstories = [ws for ws in pending_webstories if ws[0] not in processed_webstories]
            if new_webstories:
                logger.info(f"Found {len(new_webstories)} new webstories to process.")
                for webstory in new_webstories:
                    webstory_queue.put(webstory)
            else:
                logger.info(f"No new pending webstories. Waiting for 60 seconds before next check...")
                time.sleep(60)

def main():
    engine = init_connection_engine()
    backoff_time = 60  # Start with a 1-minute backoff

    while True:
        try:
            logger.info("Starting main loop...")
            pending_webstories = get_pending_webstories(engine)
            logger.info(f"Found {len(pending_webstories)} pending webstories.")
            for webstory in pending_webstories:
                webstory_queue.put(webstory)  # Put all webstory data

            process_pending_webstories(engine)
            
            # Reset backoff time if successful
            backoff_time = 60
        except Exception as e:
            logger.error(f"An error occurred in the main loop: {str(e)}")
            logger.info(f"Restarting the process in {backoff_time} seconds...")
            time.sleep(backoff_time)
            # Increase backoff time for next iteration, up to a maximum of 30 minutes
            backoff_time = min(backoff_time * 2, 1800)




if __name__ == "__main__":
    main()