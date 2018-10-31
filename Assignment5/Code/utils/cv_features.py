import numpy as np


def divide_image(image, grid_dim=(3, 3)):
    pixels = image.load()
    xSize = image.size[0]
    ySize = image.size[1]
    numPixels = int(xSize * ySize)

    width = int(xSize / grid_dim[0])
    height = int(ySize / grid_dim[1])

    flattened = flatten_image(pixels, xSize, ySize)
    grids = []
    total_pixels = 0
    for y_i in range(grid_dim[1]):
        for x_i in range(grid_dim[0]):
            pix = get_pixels(flattened,
                             range(x_i * width, (x_i + 1) * width),
                             range(y_i * height, (y_i + 1) * height), xSize)
            total_pixels += len(pix)
            grids.append(np.array(pix).reshape(8, 8))

    if not all(list(grids[-1][-1]) == flattened[-8:]):
        raise ValueError("Unexpected groups {} vs {}"
                         .format(grids[-1][-1], flattened[-8:]))
    if not total_pixels == numPixels:
        raise AssertionError("Missing pixels {} vs {}"
                             .format(numPixels, total_pixels))
    return grids


def get_pixels(flattened, xs, ys, xSize):
    """

    :param flattened: a list of pixels flattened.
    :param xs: a list of indices for x
    :param ys: a list of indices for y
    :param width: grid width
    :return: a list of pixels in flat list.
    """
    grid = []
    for y in ys:
        for x in xs:
            #index = (y * width) + x
            index = (y * xSize) + x
            grid.append(flattened[index])

    return grid


def flatten_image(pixels, xSize, ySize):
    flattened = []
    for y in range(ySize):
        for x in range(xSize):
            flattened.append(pixels[x, y])

    return np.array(flattened)


