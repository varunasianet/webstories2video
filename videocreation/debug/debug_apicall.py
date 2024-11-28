import requests
import json

def call_video_creation_api(url):
    # API endpoint
    api_url = "http://localhost:5555/create_video"

    # Request payload
    payload = {
        "url": url
    }

    # Headers
    headers = {
        "Content-Type": "application/json"
    }

    try:
        # Make the POST request
        response = requests.post(api_url, data=json.dumps(payload), headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            print("Video creation successful!")
            print(f"Vertical video URL: {data['vertical_video_url']}")
            print(f"Horizontal video URL: {data['horizontal_video_url']}")
            print(f"Execution time: {data['execution_time']}")
        else:
            print(f"Error: API returned status code {response.status_code}")
            print(f"Error message: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while making the request: {e}")

# URL to process
url = "https://newsable.asianetnews.com/webstories/entertainment/richest-bollywood-bodyguards-annual-salaries-of-top-stars-protectors-nti-sjoycq"

# Call the API
call_video_creation_api(url)
