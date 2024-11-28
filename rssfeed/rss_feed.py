import requests
import feedparser
from datetime import datetime, timezone
import json
from google.cloud.sql.connector import Connector
import sqlalchemy
import logging
import time
from sqlalchemy.dialects.mysql import insert
import pytz

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define IST timezone
ist = pytz.timezone('Asia/Kolkata')

def scrape_rss_feed(url):
    # Fetch the RSS feed
    response = requests.get(url)
    
    if response.status_code != 200:
        logger.error(f"Failed to fetch RSS feed. Status code: {response.status_code}")
        return None
    
    # Parse the RSS feed
    feed = feedparser.parse(response.content)
    
    # Extract entries
    webstories = []
    for entry in feed.entries:
        webstory_id = entry.get('id', '').split('-')[-1]  # Assuming the ID is the last part of the 'id' field
        pub_date_utc = datetime.strptime(entry.get('published', ''), '%a, %d %b %Y %H:%M:%S %z')
        pub_date_ist = pub_date_utc.astimezone(ist)
        
        webstory_data = {
            'webstory_id': webstory_id,
            'title': entry.get('title', ''),
            'link': entry.get('link', ''),
            'description': entry.get('summary', ''),
            'pubDate': pub_date_ist,
            'published_date': pub_date_ist.date(),
            'published_time': pub_date_ist.time(),
            'image': next((content['url'] for content in entry.get('media_content', []) if content.get('medium') == 'image'), ''),
            'category': json.dumps([tag.term for tag in entry.get('tags', [])]),
            'sync_time': datetime.now(ist),
            'video_creation_starttime': None,
            'video_creation_endtime': None,
            'main_bucket_path': '',
            'vertical_video_path': '',
            'horizontal_video_path': '',
            'video_available_status': 'NO',
            'content_language': 'english'  # Add this line, assuming English as default
        }
        webstories.append(webstory_data)
    
    return webstories

def init_connection_engine():
    instance_connection_name = "asianet-tech-staging:asia-south1:webstories-asianet-db"
    db_user = "webstories-user"
    db_pass = "asianetweb"
    db_name = "webstoriesrss"

    connector = Connector()

    def getconn():
        conn = connector.connect(
            instance_connection_name,
            "pymysql",
            user=db_user,
            password=db_pass,
            db=db_name,
        )
        return conn

    engine = sqlalchemy.create_engine(
        "mysql+pymysql://",
        creator=getconn,
    )
    return engine

def insert_or_update_webstories(engine, webstories):
    with engine.connect() as connection:
        metadata = sqlalchemy.MetaData()
        webstories_table = sqlalchemy.Table(
            'webstories',
            metadata,
            sqlalchemy.Column('webstory_id', sqlalchemy.String(255), primary_key=True),
            sqlalchemy.Column('title', sqlalchemy.String(255)),
            sqlalchemy.Column('link', sqlalchemy.String(255)),
            sqlalchemy.Column('description', sqlalchemy.Text),
            sqlalchemy.Column('pubDate', sqlalchemy.DateTime),
            sqlalchemy.Column('published_date', sqlalchemy.Date),
            sqlalchemy.Column('published_time', sqlalchemy.Time),
            sqlalchemy.Column('image', sqlalchemy.String(255)),
            sqlalchemy.Column('category', sqlalchemy.JSON),
            sqlalchemy.Column('sync_time', sqlalchemy.DateTime),
            sqlalchemy.Column('video_creation_starttime', sqlalchemy.DateTime),
            sqlalchemy.Column('video_creation_endtime', sqlalchemy.DateTime),
            sqlalchemy.Column('main_bucket_path', sqlalchemy.String(255)),
            sqlalchemy.Column('vertical_video_path', sqlalchemy.String(255)),
            sqlalchemy.Column('horizontal_video_path', sqlalchemy.String(255)),
            sqlalchemy.Column('video_available_status', sqlalchemy.String(3)),
            sqlalchemy.Column('content_language', sqlalchemy.String(50))
        )
        
        metadata.create_all(engine)
        
        insert_stmt = insert(webstories_table)
        on_duplicate_key_stmt = insert_stmt.on_duplicate_key_update(
            title=insert_stmt.inserted.title,
            link=insert_stmt.inserted.link,
            description=insert_stmt.inserted.description,
            pubDate=insert_stmt.inserted.pubDate,
            published_date=insert_stmt.inserted.published_date,
            published_time=insert_stmt.inserted.published_time,
            image=insert_stmt.inserted.image,
            category=insert_stmt.inserted.category,
            sync_time=insert_stmt.inserted.sync_time,
            video_creation_starttime=insert_stmt.inserted.video_creation_starttime,
            video_creation_endtime=insert_stmt.inserted.video_creation_endtime,
            main_bucket_path=insert_stmt.inserted.main_bucket_path,
            vertical_video_path=insert_stmt.inserted.vertical_video_path,
            horizontal_video_path=insert_stmt.inserted.horizontal_video_path,
            video_available_status=insert_stmt.inserted.video_available_status,
            content_language=insert_stmt.inserted.content_language
        )
        connection.execute(on_duplicate_key_stmt, webstories)
        connection.commit()

def add_missing_columns(engine):
    with engine.connect() as connection:
        inspector = sqlalchemy.inspect(engine)
        if not inspector.has_table('webstories'):
            # Create the table if it doesn't exist
            metadata = sqlalchemy.MetaData()
            webstories_table = sqlalchemy.Table(
                'webstories',
                metadata,
                sqlalchemy.Column('webstory_id', sqlalchemy.String(255), primary_key=True),
                sqlalchemy.Column('title', sqlalchemy.String(255)),
                sqlalchemy.Column('link', sqlalchemy.String(255)),
                sqlalchemy.Column('description', sqlalchemy.Text),
                sqlalchemy.Column('pubDate', sqlalchemy.DateTime),
                sqlalchemy.Column('published_date', sqlalchemy.Date),
                sqlalchemy.Column('published_time', sqlalchemy.Time),
                sqlalchemy.Column('image', sqlalchemy.String(255)),
                sqlalchemy.Column('category', sqlalchemy.JSON),
                sqlalchemy.Column('sync_time', sqlalchemy.DateTime),
                sqlalchemy.Column('video_creation_starttime', sqlalchemy.DateTime),
                sqlalchemy.Column('video_creation_endtime', sqlalchemy.DateTime),
                sqlalchemy.Column('main_bucket_path', sqlalchemy.String(255)),
                sqlalchemy.Column('vertical_video_path', sqlalchemy.String(255)),
                sqlalchemy.Column('horizontal_video_path', sqlalchemy.String(255)),
                sqlalchemy.Column('video_available_status', sqlalchemy.String(3)),
                sqlalchemy.Column('content_language', sqlalchemy.String(50))
            )
            metadata.create_all(engine)
            print("Table 'webstories' created successfully.")
        else:
            # Table exists, add missing columns
            existing_columns = set(column['name'] for column in inspector.get_columns('webstories'))
            new_columns = [
                ('video_creation_starttime', 'DATETIME'),
                ('video_creation_endtime', 'DATETIME'),
                ('main_bucket_path', 'VARCHAR(255)'),
                ('vertical_video_path', 'VARCHAR(255)'),
                ('horizontal_video_path', 'VARCHAR(255)'),
                ('video_available_status', 'VARCHAR(3)'),
                ('content_language', 'VARCHAR(50)')
            ]

            for col_name, col_type in new_columns:
                if col_name not in existing_columns:
                    alter_query = f"ALTER TABLE webstories ADD COLUMN {col_name} {col_type}"
                    connection.execute(sqlalchemy.text(alter_query))
            
            connection.commit()
            print("Missing columns added successfully.")


def main():
    # URL of the RSS feed
    rss_url = "https://newsable.asianetnews.com/rss/dh/webstories"
    
    # Initialize the database connection
    db_engine = init_connection_engine()
    
    # Add missing columns to the table
    add_missing_columns(db_engine)
    
    while True:
        try:
            # Scrape the RSS feed
            scraped_data = scrape_rss_feed(rss_url)

            if scraped_data:
                logger.info(f"Scraped {len(scraped_data)} webstories.")
                
                # Insert or update the webstories in the database
                insert_or_update_webstories(db_engine, scraped_data)
                
                logger.info("Data successfully saved to Cloud SQL database.")
            else:
                logger.warning("No data scraped from RSS feed.")
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
        
        # Wait for 5 minutes before the next check
        time.sleep(300)


if __name__ == "__main__":
    main()
