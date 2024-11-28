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
import threading
import queue
from sqlalchemy.exc import SQLAlchemyError
import argparse
import sys
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define IST timezone
ist = pytz.timezone('Asia/Kolkata')

# API endpoints
VIDEO_CREATION_API_URL = "http://localhost:5000/create_video"
VIDEO_POST_API_URL = "https://core-webservices-api.asianetnews.com/ai/video/add"

# Database connection details
INSTANCE_CONNECTION_NAME = "asianet-tech-staging:asia-south1:webstories-asianet-db"
DB_USER = "webstories-user"
DB_PASS = "asianetweb"
DB_NAME = "webstoriesrss"

# Google Cloud Storage details
BUCKET_NAME = "contentgrowth"
SERVICE_ACCOUNT_FILE = "/home/varun_saagar/videocreation/creds/asianet-tech-staging-91b6f4c817e0.json"

# Configuration
MAX_CONCURRENT_VIDEOS = 3
webstory_queue = queue.Queue()
processed_webstories = set()
video_semaphore = threading.Semaphore(MAX_CONCURRENT_VIDEOS)
video_completed_event = threading.Event()

# Request timeout
REQUEST_TIMEOUT = 300  # 5 minutes

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
            SELECT webstory_id, link, vertical_video_url, horizontal_video_url, posted_to_api
            FROM webstories
            WHERE (posted_to_api = 'NO' OR posted_to_api IS NULL)
               OR (vertical_video_url IS NULL OR horizontal_video_url IS NULL)
            ORDER BY sync_time DESC
        """)
        result = execute_with_retry(connection, query)
        return result.fetchall()

def update_webstory_status(engine, webstory_id, vertical_path, horizontal_path, vertical_video_url, horizontal_video_url, status):
    max_length = 255
    with engine.begin() as connection:
        query = sqlalchemy.text("""
            UPDATE webstories
            SET video_creation_starttime = :start_time,
                video_creation_endtime = :end_time,
                vertical_video_path = :vertical_path,
                horizontal_video_path = :horizontal_path,
                vertical_video_url = :vertical_video_url,
                horizontal_video_url = :horizontal_video_url,
                video_available_status = :status
            WHERE webstory_id = :webstory_id
        """)
        params = {
            "start_time": datetime.now(ist),
            "end_time": datetime.now(ist),
            "vertical_path": vertical_path[:max_length] if vertical_path else None,
            "horizontal_path": horizontal_path[:max_length] if horizontal_path else None,
            "vertical_video_url": vertical_video_url[:max_length] if vertical_video_url else None,
            "horizontal_video_url": horizontal_video_url[:max_length] if horizontal_video_url else None,
            "status": status,
            "webstory_id": webstory_id
        }
        execute_with_retry(connection, query, params)

def update_posted_to_api_status(engine, webstory_id, status):
    with engine.begin() as connection:
        query = sqlalchemy.text("""
            UPDATE webstories
            SET posted_to_api = :status
            WHERE webstory_id = :webstory_id
        """)
        execute_with_retry(connection, query, {"status": status, "webstory_id": webstory_id})

def create_video(url, max_retries=3, retry_delay=30):
    for attempt in range(max_retries):
        try:
            logger.info(f"Sending request to create video for URL: {url} (Attempt {attempt + 1})")
            response = requests.post(VIDEO_CREATION_API_URL, json={"url": url}, headers={"Content-Type": "application/json"}, timeout=REQUEST_TIMEOUT)
            logger.info(f"Response status code: {response.status_code}")
            logger.info(f"Response content: {response.text}")
            response.raise_for_status()
            
            result = response.json()
            if "vertical_video_url" in result and "horizontal_video_url" in result:
                logger.info(f"Video creation successful for URL: {url}")
                return result
            else:
                logger.error(f"Unexpected response format for video creation: {result}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"An error occurred while creating video for URL: {url}: {e}")
        
        if attempt < max_retries - 1:
            logger.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            logger.error(f"Max retries reached for URL: {url}")
            return None

def post_video_urls(engine, webstory_id, article_url, vertical_video_url, horizontal_video_url):
    headers = {"Content-Type": "application/json"}
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
class VideoWorker(threading.Thread):
    def __init__(self, worker_id, post_to_api):
        super().__init__()
        self.daemon = True
        self.worker_id = worker_id
        self.logger = logging.getLogger(f"VideoWorker-{worker_id}")
        self.post_to_api = post_to_api

    def run(self):
        engine = init_connection_engine()
        while True:
            try:
                with video_semaphore:
                    self.logger.info(f"Worker {self.worker_id} waiting for a webstory...")
                    webstory = webstory_queue.get(block=True, timeout=60)
                    if webstory is None:
                        self.logger.info(f"Worker {self.worker_id} shutting down")
                        webstory_queue.task_done()
                        return

                    webstory_id, link, vertical_video_url, horizontal_video_url, posted_to_api = webstory
                    
                    self.logger.info(f"Worker {self.worker_id} processing webstory ID: {webstory_id}")
                
                    # Check if we need to generate the video
                    if not vertical_video_url or not horizontal_video_url:
                        self.logger.info(f"Generating video for webstory ID: {webstory_id}")
                        result = create_video(link)
                        
                        if result and "vertical_video_url" in result and "horizontal_video_url" in result:
                            vertical_video_url = result["vertical_video_url"]
                            horizontal_video_url = result["horizontal_video_url"]
                            
                            self.logger.info(f"Updating database with new video URLs for webstory ID: {webstory_id}")
                            update_webstory_status(engine, webstory_id, vertical_video_url, horizontal_video_url, vertical_video_url, horizontal_video_url, 'YES')
                        else:
                            self.logger.warning(f"Failed to create video for webstory ID: {webstory_id}")
                            update_webstory_status(engine, webstory_id, '', '', '', '', 'FAILED')
                            continue
                    
                    # Post video URLs if they haven't been posted before and post_to_api is True
                    if self.post_to_api and posted_to_api != 'YES':
                        self.logger.info(f"Posting video URLs for webstory ID: {webstory_id}")
                        if post_video_urls(engine, webstory_id, link, vertical_video_url, horizontal_video_url):
                            self.logger.info(f"Video URLs posted for webstory ID: {webstory_id}")
                        else:
                            self.logger.warning(f"Failed to post video URLs for webstory ID: {webstory_id}")
                    elif not self.post_to_api:
                        self.logger.info(f"Skipping API post for webstory ID: {webstory_id} as post_to_api is set to False")
                    else:
                        self.logger.info(f"Video URLs for webstory ID: {webstory_id} have already been posted. Skipping.")
                    
                    self.logger.info(f"Worker {self.worker_id} completed processing webstory ID: {webstory_id}")
                    processed_webstories.add(webstory_id)
                    webstory_queue.task_done()

                    # Signal that a video is completed
                    video_completed_event.set()

            except queue.Empty:
                self.logger.info(f"Worker {self.worker_id} timed out waiting for a webstory. Continuing...")
            except Exception as e:
                self.logger.error(f"Error processing webstory: {str(e)}")
                self.logger.debug(traceback.format_exc())
            finally:
                if 'webstory' in locals():
                    webstory_queue.task_done()

def run_workers(post_to_api):
    workers = []
    for i in range(3):  # Create 3 workers
        worker = VideoWorker(i, post_to_api)
        worker.start()
        workers.append(worker)
    return workers

def check_api_health():
    try:
        response = requests.get(VIDEO_CREATION_API_URL.rsplit('/', 1)[0] + '/health', timeout=10)
        if response.status_code == 200:
            logger.info("API health check passed")
            return True
        else:
            logger.error(f"API health check failed. Status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"API health check failed. Error: {e}")
        return False

def parse_args():
    parser = argparse.ArgumentParser(description="Video creation and posting script")
    parser.add_argument("--post_to_api", choices=['yes', 'no'], default='no',
                        help="Whether to post video URLs to API (yes) or just generate videos (no)")
    return parser.parse_args()

def main():
    args = parse_args()
    post_to_api = args.post_to_api == 'yes'

    if not check_api_health():
        logger.critical("Video creation API is not accessible. Please check if it's running and try again.")
        return

    engine = init_connection_engine()
    workers = run_workers(post_to_api)

    try:
        while True:
            pending_webstories = get_pending_webstories(engine)
            new_webstories = [ws for ws in pending_webstories if ws[0] not in processed_webstories]
            
            logger.info(f"Found {len(new_webstories)} new webstories to process")
            
            for ws in new_webstories:
                logger.info(f"Adding webstory ID {ws[0]} to the queue")
                webstory_queue.put(ws)
            
            if not new_webstories:
                logger.info("No new pending webstories. Waiting...")
                time.sleep(60)
                continue

            # Wait for a video to complete or timeout after 5 minutes
            if not video_completed_event.wait(timeout=2000):
                logger.warning("Timeout waiting for video completion. Continuing...")
            video_completed_event.clear()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down...")
    except Exception as e:
        logger.critical(f"Unhandled exception in main loop: {str(e)}")
        logger.critical(traceback.format_exc())
    finally:
        logger.info("Shutting down workers...")
        for _ in workers:
            webstory_queue.put(None)
        
        for worker in workers:
            worker.join(timeout=10)
            if worker.is_alive():
                logger.warning(f"Worker {worker.worker_id} did not shut down gracefully")
        
        logger.info("All worker threads shut down.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {str(e)}")
        logger.critical(traceback.format_exc())
        sys.exit(1)
