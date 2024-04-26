# Serve flask app from ubuntu
FROM jjanzic/docker-python3-opencv

# Install dependencies
RUN pip3 install flask gunicorn

# Copy files
WORKDIR /app

COPY . /app

# Run with gunicorn
CMD ["gunicorn", "-w 12", "--bind", "0.0.0.0:80", "app:app"]