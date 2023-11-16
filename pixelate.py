#!/usr/bin/env python3

# This project will read an input image and automatically pixelate it and generate a new image.
import argparse
import os
import cv2
import numpy as np

DEBUG = False

# Parse command line arguments
parser = argparse.ArgumentParser(description="Pixelate an image.")
parser.add_argument("-o", "--output", type=str, required=False, help="Output directory")
parser.add_argument(
    "-c", "--colors", type=int, required=False, default=128, help="Number of colors"
)
parser.add_argument(
    "-s", "--size", type=int, required=False, default=1, help="Pixel size"
)
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Enable verbose output"
)
parser.add_argument(
    "input_files", nargs="+", metavar="FILE", help="Input files to process"
)

args = parser.parse_args()

for image in args.input_files:
    # Read input image
    input_image = cv2.imread(image)
    if DEBUG:
        cv2.imshow("Original", input_image)
        cv2.waitKey(0)

    # Downscale image
    downscale = 1 / args.size
    scaled_image = cv2.resize(input_image, (0, 0), fx=downscale, fy=downscale)

    # Now I use k-means to cluster the colors, and then replace each pixel with the centroid of its cluster
    # Reshape image to be just a list of pixels, each pixel is a 3-element vector of floats
    pixels = scaled_image.reshape((scaled_image.shape[0] * scaled_image.shape[1], 3))
    pixels = np.float32(pixels)

    # Run k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(
        pixels, args.colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # Convert back to 8 bit values
    center = np.uint8(center)

    # Replace each pixel value with its center
    res = center[label.flatten()]
    res2 = res.reshape((scaled_image.shape))

    # Count the number of pixels of each color
    color_counts = {}
    for i in range(len(res)):
        color = tuple(res[i])
        if color in color_counts:
            color_counts[color] += 1
        else:
            color_counts[color] = 1
    order = sorted(color_counts, reverse=True)

    # Later, I require the image to be a power of 2 in each dimension, so I pad it here with the most common color
    padding = np.uint(2 ** np.ceil(np.log2(max(res2.shape))))

    # padded_image is the end goal
    padded_image = np.zeros((padding, padding, 3), np.uint8)
    padded_image[:, :] = order[0]
    padded_image[: res2.shape[0], : res2.shape[1]] = res2

    final_image = np.zeros((padding, padding, 3), np.uint8)
    bitmask = np.zeros((padding, padding), bool)

    # Now it's time for quad-tree decomposition
    instructions = {}
    for color in order:
        instructions[color] = []

    # The first color just gets the whole image
    instructions[order[0]] = [(0, 0, res2.shape[1], res2.shape[0])]

    # Fill the bitmask with True for each pixel that is the common color
    bitmask[padded_image[:, :, 0] == order[0][0]] = True
    final_image[bitmask] = order[0]

    # Show the bitmask and image
    if DEBUG:
        cv2.imshow(
            "Image " + str(order[0]),
            cv2.resize(
                final_image, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST
            ),
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Now we do quad-tree decomposition for the rest of the colors
    for color in order[1:]:
        
        def quad_tree(x, y, w, h):
            # If this color is no longer in the set of pixels we're looking at, return
            if not np.any(padded_image[y : y + h, x : x + w] == color):
                return

            # If each pixel is this color OR it's not in the bitmask
            # Create 2 bitmask and OR them together
            colored = np.all(padded_image[y : y + h, x : x + w] == color, axis=-1)
            mask = bitmask[y : y + h, x : x + w] == False

            if np.all(np.logical_or(colored, mask)):
                # Fill the region
                final_image[y : y + h, x : x + w] = color
                instructions[color].append((x, y, w, h))
                # Add to the bitmask the colored pixels
                bitmask[y : y + h, x : x + w] = colored
            else:
                # Otherwise, split it into 4 quadrants and recurse
                quad_tree(x, y, w // 2, h // 2)
                quad_tree(x + w // 2, y, w // 2, h // 2)
                quad_tree(x, y + h // 2, w // 2, h // 2)
                quad_tree(x + w // 2, y + h // 2, w // 2, h // 2)

        quad_tree(0, 0, padded_image.shape[0], padded_image.shape[1])

        # Show the bitmask and image
        if DEBUG:
            cv2.imshow(
                "Image " + str(color),
                cv2.resize(
                    final_image, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST
                ),
            )
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Show what the final image should look like (undo the padding)
    if DEBUG:
        cv2.imshow(
            "Final",
            cv2.resize(
                final_image[: res2.shape[0], : res2.shape[1]],
                (0, 0),
                fx=4,
                fy=4,
                interpolation=cv2.INTER_NEAREST,
            ),
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Scale the instructions back up to the original size
    for color in order:
        instructions[color] = [
            (x * args.size, y * args.size, w * args.size, h * args.size)
            for (x, y, w, h) in instructions[color]
        ]

        # Sort our instructions by x, then y
        instructions[color] = sorted(instructions[color], key=lambda x: (x[1], x[0]))

        # Merge adjacent rectangles
        for i in range(len(instructions[color]) - 1, 0, -1):
            (x1, y1, w1, h1) = instructions[color][i]
            (x2, y2, w2, h2) = instructions[color][i - 1]
            if y1 == y2 and h1 == h2 and x1 == x2 + w2:
                # If the y and height are the same, merge them
                instructions[color][i - 1] = (x2, y2, w1 + w2, h1)
                del instructions[color][i]

        # Sort the other way (top to bottom)
        instructions[color] = sorted(instructions[color], key=lambda x: (x[0], x[1]))

        # Merge adjacent rectangles
        for i in range(len(instructions[color]) - 1, 0, -1):
            (x1, y1, w1, h1) = instructions[color][i]
            (x2, y2, w2, h2) = instructions[color][i - 1]
            if x1 == x2 and w1 == w2 and y1 == y2 + h2:
                # If the x and width are the same, merge them
                instructions[color][i - 1] = (x2, y2, w1, h1 + h2)
                del instructions[color][i]

    # Now we output the instructions to a js file
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    file = args.output + "/" + os.path.basename(image) + ".js"
    with open(file, "w") as f:
        f.write("/// THIS FILE WAS COMPUTER GENERATED\n")
        f.write("function myDraw() {\n")
        for color in order:
            f.write("\tfill({0}, {1}, {2});\n".format(color[2], color[1], color[0]))
            for instruction in instructions[color]:
                f.write(
                    "\trect({0}, {1}, {2}, {3});\n".format(
                        instruction[0], instruction[1], instruction[2], instruction[3]
                    )
                )
            f.write("\n")
        f.write("}\n")

        f.write(
            """
function setup() {{
    createCanvas({0}, {1});
    background(0);
    noStroke();
    myDraw();
}}""".format(
                instructions[order[0]][0][2], instructions[order[0]][0][3]
            )
        )
        f.write(
            """
var strokeState = false;
function keyPressed() {{
    if (keyCode == BACKSPACE) {{
        strokeState = !strokeState;
        if (strokeState) {{
            stroke(0, 0, 0);
        }} else {{
            noStroke();
        }}
        myDraw();
    }}
}}"""
        )

    print(file)
