#!/usr/bin/env python3

# This is a simple server that reads javascript code from a directory, and makes it into p5.js sketches.

import argparse
import os

from flask import Flask, render_template, send_from_directory

app = Flask(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Pixelate an image.")
parser.add_argument("-p", "--port", type=int, required=False, default=4444, help="Port to listen on")

args = parser.parse_args()

# Serve static files
@app.route("/static/<path:path>")
def send_static(path):
    return send_from_directory("static", path)

# Serve sketches
@app.route("/sketches/<path:path>")
def send_sketch(path):
    return send_from_directory("sketches", path)

# Serve viewer
@app.route("/view/<path:path>")
def send_viewer(path):
    # Parses the view.html template, adding the filename to the template
    return render_template('view.html', title=path)

# Serve index
@app.route("/")
def send_index():
    projects = [project[:-3] for project in os.listdir('sketches')]
    print(projects)
    return render_template('index.html', projects=projects)

# Run the server
app.run(host="127.0.0.0", port=args.port, debug=True)