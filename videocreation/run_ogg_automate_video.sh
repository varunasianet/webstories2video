#!/bin/bash

# Function to get current date in YYYY-MM-DD format
get_current_date() {
    echo $(date +%Y-%m-%d)
}

# Function to create log directory if it doesn't exist
create_log_directory() {
    local log_dir="logs"
    if [ ! -d "$log_dir" ]; then
        mkdir -p "$log_dir"
    fi
}

# Function to rotate logs (delete logs older than 7 days)
rotate_logs() {
    local log_dir="logs"
    find "$log_dir" -type f -mtime +7 -delete
}

# Main loop
while true; do
    # Get current date
    current_date=$(get_current_date)

    # Create log directory if it doesn't exist
    create_log_directory

    # Rotate logs (delete old ones)
    rotate_logs

    # Run the Python script and redirect output to dated log file
    python3 ogg_automate_video.py > "logs/outputfixcrdmoct1_ogg_automate_video_$current_date.log" 2>&1 &
    
    # Sleep for 24 hours
    sleep 86400
    
    # Kill the Python process
    pkill -f ogg_automate_video.py
    
    # Wait for cleanup
    sleep 60
done
