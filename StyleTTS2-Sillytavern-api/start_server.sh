#!/bin/bash

# Activate the virtual environment
source /home/varun_saagar/StyleTTS2-Sillytavern-api/styletts/bin/activate

# Print Python version and path for debugging
echo "Python version:"
python --version
echo "Python path:"
which python

# Print Gunicorn version for debugging
echo "Gunicorn version:"
gunicorn --version

# Change to the correct directory
cd /home/varun_saagar/StyleTTS2-Sillytavern-api

exec gunicorn --bind 0.0.0.0:8001 --log-level debug --error-logfile /home/varun_saagar/gunicorn-error.log wsgi:app 2>&1 | tee /home/varun_saagar/gunicorn-output.log
