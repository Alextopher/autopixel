#!/usr/bin/env python3

# This project will read an input image and automatically pixelate it and generate a new image.
import argparse
import os
import cv2
import numpy as np

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
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug output")
parser.add_argument(
    "--no-optimize", dest="optimize", action="store_false", help="Disable optimization"
)
parser.add_argument(
    "input_files", nargs="+", metavar="FILE", help="Input files to process"
)

args = parser.parse_args()


# debug shows an image at each step of the process
def debug(image: np.ndarray, name: str, scale=1):
    if args.debug:
        cv2.imshow(
            name,
            cv2.resize(
                image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
            ),
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Down samples the image by a factor of 'size'.
def downscale(image, size):
    # Downscale image
    downscale = 1 / size
    scaled_image = cv2.resize(image, (0, 0), fx=downscale, fy=downscale)
    return scaled_image


# An image typically has _many_ colors, but to make our pixel art look "pixelated"
# we want to reduce the number of colors, in addition to reducing the resolution.
#
# This function does that.
#
# It uses a machine learning (AI) algorithm called k-means clustering to find the
# most effective colors to use.
def reduce_colors(image: np.ndarray, num_colors: int):
    # Flattens the image from a 2D array of pixels to a 1D array (list) of pixels.
    pixels = image.reshape((image.shape[0] * image.shape[1], 3))

    # Our RGB values are currently 8-bit integers, but to use k-means clustering
    # we need decimal values, here I convert the values to 32-bit floats.
    #
    # Eg. [0, 10, 255] -> [0.0, 10.0, 255.0]
    pixels = np.float32(pixels)

    # Criteria here is to run the k-means clustering algorithm until we're
    # within 1.0 color of the actual color.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, colors = cv2.kmeans(
        pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # Colors is a list of best colors to use, it is length num_colors. Bring it
    # from 32-bit floats back to 8-bit integers
    colors = np.uint8(colors)

    # This is a tiny quirk in numpy, labels is a 2D array [pixels, 1], but I
    # want it to be a 1D array [pixels]. This does that.
    labels = labels.flatten()

    # Labels is a list of _indexes_ into colors, it is like
    # [0, 1, 2, 1, 4, 0, ...]
    # Here I convert it to a list of _colors_ instead of indexes
    updated_pixels = np.take(colors, labels, axis=0)
    # Reshape the list of pixels back to a 2D array
    updated_image = updated_pixels.reshape(image.shape)

    # In addition to trimming the number of colors, I also want to return the
    # order of the colors, from most common to least common.
    unique_pixels, counts = np.unique(updated_pixels, return_counts=True, axis=0)
    sorted_colors = unique_pixels[np.argsort(-counts)]

    # Later, I need to put the colors into a dictionary, but numpy arrays can't
    # be keys in dictionaries, so I convert them to tuples.
    sorted_colors = [tuple(color) for color in sorted_colors]

    return (updated_image, sorted_colors)


# Pads an image so that each dimension is a power of 2.
#
# My "quad-tree decomposition" algorithm requires that the image a square of
# with power of 2 length
#
# Any extra space is filled with the given color.
def pad(image: np.ndarray, color):
    (w, h) = image.shape[:2]
    x_padding = int(pow(2, np.ceil(np.log2(w))))
    y_padding = int(pow(2, np.ceil(np.log2(h))))

    padding = max(x_padding, y_padding)

    padded_image = np.full((padding, padding, 3), color, np.uint8)
    padded_image[:w, :h] = image

    return padded_image


# Preforms quad-tree decomposition on the image, for each color.
#
# It takes the input image, and the order of colors to use, and returns a list
# of rectangles to draw, for each color.
def decompose(image: np.ndarray, colors: list):
    # assert that the image is square, and a power of 2 size
    assert image.shape[0] == image.shape[1]
    assert image.shape[0] & (image.shape[0] - 1) == 0

    # This bitmask keeps track of which pixels have been given their
    # _final_ color. While building the image, pixels may be given a temporary
    # color, allowing us to paint larger regions.
    #
    # This serves as an important optimization, the resulting program will have
    # much fewer rectangles to draw.
    bitmask = np.full(image.shape[:2], False, bool)

    # if debug mode is enabled I want to track the progress of the algorithm here
    final_image = None
    if args.debug:
        final_image = np.full(image.shape, colors[0], np.uint8)

    # Lists of rectangles to draw, for each color.
    instructions = []

    # The first color gets the whole image!
    instructions.append([(0, 0, image.shape[1], image.shape[0])])

    # Update the bitmask, marking all pixels that are the first color as True
    bitmask = np.all(image == colors[0], axis=-1)

    # Now for the rest of the colors
    for color in colors[1:]:
        rectangles = []

        # work is a stack of rectangles to process, it starts with the whole image
        # and is split into smaller rectangles as the algorithms progresses.
        #
        # Each rectangle is a tuple of (x, y, width, height)
        work = [(0, 0, image.shape[1], image.shape[0])]

        # While there is still work to do
        while len(work) > 0:
            (x, y, w, h) = work.pop()
            region = image[y : y + h, x : x + w]

            # If this color is no longer in the region, skip it
            if not np.any(region == color):
                continue

            # Otherwise, we check if the region is allowed to be fully colored
            if np.all(bitmask[y : y + h, x : x + w] == False):
                rectangles.append((x, y, w, h))

                # Update the bitmask, marking all pixels that are this color as True
                bitmask[y : y + h, x : x + w] = np.all(region == color, axis=-1)

                if args.debug:
                    final_image[y : y + h, x : x + w] = color
            else:
                # Otherwise, split it into 4 quadrants and add them to the work list
                work.append((x, y, w // 2, h // 2))
                work.append((x + w // 2, y, w // 2, h // 2))
                work.append((x, y + h // 2, w // 2, h // 2))
                work.append((x + w // 2, y + h // 2, w // 2, h // 2))

        instructions.append(rectangles)

        if args.debug:
            debug(final_image, "Color: " + str(color))

    return instructions


# Optimizes the program by merging adjacent rectangles.
#
# This is an optional step, but it can make major improvements to the resulting
# program.
#
# This edits the program in place.
def optimize(program):
    optimized = []

    for instructions in program:
        # Sort left to right
        instructions = sorted(instructions, key=lambda x: (x[1], x[0]))

        # Merge adjacent rectangles
        for i in range(len(instructions) - 1, 0, -1):
            (x1, y1, w1, h1) = instructions[i]
            (x2, y2, w2, h2) = instructions[i - 1]
            if y1 == y2 and h1 == h2 and x1 == x2 + w2:
                # If the y and height are the same, merge them
                instructions[i - 1] = (x2, y2, w1 + w2, h1)
                del instructions[i]

        # Sort the other way (top to bottom)
        instructions = sorted(instructions, key=lambda x: (x[0], x[1]))

        # Merge adjacent rectangles
        for i in range(len(instructions) - 1, 0, -1):
            (x1, y1, w1, h1) = instructions[i]
            (x2, y2, w2, h2) = instructions[i - 1]
            if x1 == x2 and w1 == w2 and y1 == y2 + h2:
                # If the x and width are the same, merge them
                instructions[i - 1] = (x2, y2, w1, h1 + h2)
                del instructions[i]

        optimized.append(instructions)

    return optimized


# Repeatidly preforms single-pass optimizations until the program stops changing.
def optimize_until_stable(program):
    size = sum([len(instructions) for instructions in program])
    while True:
        program = optimize(program)
        new_size = sum([len(instructions) for instructions in program])
        if new_size == size:
            break
        size = new_size
    return program


# Saves the program to a file
def save(colors, program, dim, file):
    file.write("/// THIS FILE WAS COMPUTER GENERATED\n")
    file.write("function myDraw() {\n")
    for i, color in enumerate(colors):
        file.write("\tfill({0}, {1}, {2});\n".format(color[2], color[1], color[0]))
        for instruction in program[i]:
            file.write(
                "\trect({0}, {1}, {2}, {3});\n".format(
                    instruction[0], instruction[1], instruction[2], instruction[3]
                )
            )
        file.write("\n")
    file.write("}\n")

    file.write("function setup() {\n")
    file.write("\tcreateCanvas({0}, {1});\n".format(w, h))
    file.write("\tbackground(0);\n")
    file.write("\tnoStroke();\n")
    file.write("\tmyDraw();\n")
    file.write("}\n")

    file.write("var strokeState = false;\n")
    file.write("function keyPressed() {\n")
    file.write("\tif (keyCode == BACKSPACE) {\n")
    file.write("\t\tstrokeState = !strokeState;\n")
    file.write("\t\tif (strokeState) {\n")
    file.write("\t\t\tstroke(0, 0, 0);\n")
    file.write("\t\t} else {\n")
    file.write("\t\t\tnoStroke();\n")
    file.write("\t\t}\n")
    file.write("\t\tmyDraw();\n")
    file.write("\t}\n")
    file.write("}\n")


# Main program sequence!
for image in args.input_files:
    # Add timing information
    if args.verbose:
        import time

        start = time.time()

    if args.verbose:
        print(f"Processing {image}...")

    # Read input image
    input_image = cv2.imread(image)
    (h, w) = input_image.shape[:2]

    # Shrink the image (pixelate it)
    scaled_image = downscale(input_image, args.size)
    if args.debug:
        debug(scaled_image, "Scaled")

    # Chooses best-fit colors
    (reduced, colors) = reduce_colors(scaled_image, args.colors)
    if args.debug:
        debug(reduced, "Reduced")

    # Makes the image a square, with power of 2 sides.
    padded_image = pad(reduced, colors[0])

    if args.debug:
        debug(padded_image, "Padded")

    # Decompose the image for each color
    program = decompose(padded_image, colors)

    # Scale the programs instructions back up to the original size
    def scale(instruction):
        (x, y, w, h) = instruction
        return (x * args.size, y * args.size, w * args.size, h * args.size)

    program = [
        [scale(instruction) for instruction in instructions] for instructions in program
    ]

    if args.verbose:
        # Print the original program size before optimizing
        print(
            f"Original program size: {sum([len(instructions) for instructions in program])}"
        )

    # Optimize the program, joining adjacent rectangles
    if args.optimize:
        program = optimize(program)

        if args.verbose:
            # Print the optimized program size
            print(
                f"Optimized program size: {sum([len(instructions) for instructions in program])}"
            )

    # Now we output the instructions to a js file
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    output_file = os.path.join(args.output, os.path.basename(image) + ".js")

    if args.verbose:
        print(f"Writing to {output_file}...")

    with open(output_file, "w") as file:
        save(colors, program, (w, h), file)

    if args.verbose:
        end = time.time()
        print(f"Done in {round(end - start, 2)} seconds.")
