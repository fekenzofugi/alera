from multiprocessing import Process
from app import create_app

app = create_app()

if __name__ == "__main__":
    # Start Flask app using Gunicorn in a separate process
    flask_process = Process(target=app.run, kwargs={"host": "localhost", "port": 5000, "debug": True})    # Start the processes
    flask_process.start()
    # Wait for the processes to complete
    flask_process.join()