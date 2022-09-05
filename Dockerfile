# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim

# Allow statements and log messages to immediately appear in the Knative logs
# ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
WORKDIR /app
COPY . ./

# Install production dependencies.
RUN pip install -r requirements.txt
EXPOSE 8000

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
# ENV FLASK-APP ./src/app.py
# CMD [ "python", "app.py" ]
# CMD exec python -m flask
# with gunicorn
CMD exec gunicorn --workers 1 app:app
# CMD exec gunicorn --bind :8080 --workers 1 app:app