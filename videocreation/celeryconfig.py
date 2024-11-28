# celeryconfig.py

broker_url = 'redis://localhost:6379/0'
result_backend = 'redis://localhost:6379/0'

task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
timezone = 'Asia/Kolkata'
enable_utc = False

worker_max_tasks_per_child = 1000
worker_max_memory_per_child = 200000  # 200MB
worker_shutdown_timeout = 300  # 5 minutes

task_time_limit = 1800  # 30 minutes
task_soft_time_limit = 1700  # 28 minutes

imports = ('tasks',)  # Add your task modules here
