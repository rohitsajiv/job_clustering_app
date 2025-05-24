import schedule
import time
from app import run_job_pipeline

def job():
    print("Running scheduled job pipeline...")
    run_job_pipeline()

def start_scheduler():
    schedule.every(6).hours.do(job)  # Change interval as needed

    while True:
        schedule.run_pending()
        time.sleep(60)  # check every minute
