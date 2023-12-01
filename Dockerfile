# Serve flask app from ubuntu
FROM ubuntu:latest

# Install python3 and pip3
RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential

# Install dependencies
RUN apt-get install python3-opencv
RUN pip3 install flask numpy gunicorn

# Copy files
WORKDIR /app

COPY . /app

# Run with gunicorn
CMD ["gunicorn", "-w 2", "--bind", "0.0.0.0:80", "app:app"]