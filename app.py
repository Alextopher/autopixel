#!/usr/bin/env python3

import argparse
import cv2
import numpy as np

from flask import Flask, redirect, render_template, request, send_from_directory

from pixelate import pixelate

app = Flask(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Pixelate an image.")
parser.add_argument(
    "-p", "--port", type=int, required=False, default=4444, help="Port to listen on"
)

args = parser.parse_args()

# Upload images (in addition to an image it requires number of colors and pixel size)
#
# We need to save both the javascript and the output image to the sketches directory
# Then this function should return the url to those 2 files
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    num_colors = request.form.get("colors")
    pixel_size = request.form.get("size")

    # Validate input
    if not file:
        return "No image uploaded", 400

    if not num_colors:
        return "Number of colors not specified", 400
    
    if not pixel_size:
        return "Pixel size not specified", 400

    # Hash the image to get a unique filename
    filename = str(hash(file)) + "_" + num_colors + "_" + pixel_size

    # read image as an numpy array 
    image = np.asarray(bytearray(file.read()), dtype="uint8") 
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    cv2.imwrite(f"sketches/{filename}_original.png", image)
      
    # # Process image
    javascript, output_image = pixelate(image, int(num_colors), int(pixel_size))

    # # Save javascript and output image to sketches directory
    with open(f"sketches/{filename}.js", "w") as f:
        f.write(javascript)

    cv2.imwrite(f"sketches/{filename}.png", output_image)

    # # Redirect to the viewer page
    return redirect(f"/view/{filename}")


# Serve the index page
@app.route("/")
def index():
    return render_template("index.html")

# Serve the viewer page
@app.route("/view/<filename>")
def view(filename):
    return render_template("view.html", filename=filename)


# Serve static files from the static directory
@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory("static", path)

# Serve the output image from the sketches directory
@app.route("/sketches/<path:path>")
def sketches(path):
    return send_from_directory("sketches", path)

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port, debug=True)