import numpy as np


def divide_image(image, grid_dim=(3, 3)):
    """
    Divide image into N x M grid
    :param image:
    :param grid_dim:
    :return: a list of numpy arrays

    For example, for 24 x 24 pixel image, if griding it into 3 x 3,
    this function returns a list of arrays below.

    | ------------- | --------------| ------------- |
    | 3 x 3 Array 1 | 3 x 3 Array 2 | 3 x 3 Array 3 |
    | ------------- | ----------- --| ------------- |
    | 3 x 3 Array 4 | 3 x 3 Array 5 | 3 x 3 Array 6 |
    | ------------- | ----------- --| ------------- |
    | 3 x 3 Array 7 | 3 x 3 Array 8 | 3 x 3 Array 9 |
    | ------------- | --------------| ------------- |

    returns...
    | ------------- | --------------|...|...| ------------- |
    | 3 x 3 Array 1 | 3 x 3 Array 2 |...|...| 3 x 3 Array 9 |
    | ------------- | ----------- --|...|...| ------------- |

    """
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
            index = (y * xSize) + x
            grid.append(flattened[index])

    return grid


def flatten_image(pixels, xSize, ySize):
    flattened = []
    for y in range(ySize):
        for x in range(xSize):
            flattened.append(pixels[x, y])

    return np.array(flattened)


def get_x_gradients(grid):

    columns, rows = grid.shape
    gradients = []
    for row in range(rows):
        intensities = [c / 255. for c in grid[row, :]]
        for i, column in enumerate(intensities):
            if i == 0:
                gradients.append(intensities[1] - intensities[0])
            elif i == (columns - 1):
                gradients.append(intensities[i] - intensities[i - 1])
            else:
                gradients.append(intensities[i + 1] - intensities[i - 1])

    return gradients


def get_x_gradient_features(image, grid_dim=(3, 3)):
    """

    :param image: an instance of PIL.Image.Image object.
    :param grid_dim:
    :return: a list of gradient min, max, average features of each grid.
    """
    grids = divide_image(image, grid_dim)
    features = []
    for grid in grids:
        gradients = get_x_gradients(grid)
        features.append(min(gradients))
        features.append(max(gradients))
        features.append(sum(gradients) / len(gradients))

    return features


def get_y_gradient_features(image, grid_dim=(3, 3)):
    """

    :param image: an instance of PIL.Image.Image object.
    :param grid_dim:
    :return: a list of gradient min, max, average features of each grid.
    """
    image = image.rotate(90)
    return get_x_gradient_features(image, grid_dim)






