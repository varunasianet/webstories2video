
# Webstories2Video

An automated system that converts web stories into video content using RSS feeds, text-to-speech, and AI-powered video generation.

## Description
This project combines three main components:
- RSS Feed Parser: Extracts content from web stories
- Video Creation: Converts text and images into video format
- StyleTTS2: Provides high-quality text-to-speech conversion

The system automates the process of converting web stories into engaging video content, making it easier to repurpose content across different platforms.

## Features
- RSS feed parsing and content extraction
- Automated video generation from text and images
- AI-powered text-to-speech conversion
- Support for multiple story formats
- Configurable video output settings

## Requirements
- Python 3.8+
- FFmpeg
- Required Python packages (listed in requirements.txt)
- GPU support (recommended for StyleTTS2)

## Installation
1. Clone the repository:
```bash
git clone https://github.com/varunasianet/webstories2video.git
cd webstories2video
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up the required environment variables:
```bash
# Configure your environment variables here
```

## Usage
1. Configure RSS feed sources in `config.json`
2. Run the RSS feed parser:
```bash
python rssfeed/main.py
```

3. Generate videos:
```bash
python videocreation/app.py
```

## Project Structure
```
webstories2video/
├── rssfeed/           # RSS feed parsing module
├── videocreation/     # Video generation module
├── StyleTTS2/         # Text-to-speech module
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```
