import os
import unittest

import PIL
from Assignment5.Code import kDataPath
from PIL import Image
from utils.cv_features import divide_image, get_x_gradients, get_x_gradient_features, get_y_gradient_features


class TestCVFeatures(unittest.TestCase):


    def test_grid_image(self):
        fname = os.path.join(kDataPath, r'closedRightEyes/closed_eye_0001.jpg_face_1_R.jpg')
        image = Image.open(fname, 'r')
        grids = divide_image(image, grid_dim=(3, 3))
        self.assertTrue(len(grids) == 9)

        from matplotlib import pyplot as plt
        for i in range(1, 10):
            plt.subplot(3, 3, i)
            plt.gray()
            plt.imshow(grids[i - 1])
        plt.show()

    def test_get_x_gradients(self):
        fname = os.path.join(kDataPath, r'closedRightEyes/closed_eye_0001.jpg_face_1_R.jpg')
        image = Image.open(fname, 'r')
        grids = divide_image(image, grid_dim=(3, 3))
        for grid in grids:
            x_grads = get_x_gradients(grid)
            self.assertTrue(len(x_grads) == 64)

        features = get_x_gradient_features(image, grid_dim=(3, 3))
        self.assertTrue(len(features) == 27)

        features = get_y_gradient_features(image, grid_dim=(3, 3))
        self.assertTrue(len(features) == 27)

    def test_divide_image_1x1(self):
        fname = os.path.join(kDataPath, r'closedRightEyes/closed_eye_0001.jpg_face_1_R.jpg')
        image = Image.open(fname, 'r')
        grids = divide_image(image, grid_dim=(1, 1))
        self.assertTrue(len(grids) == 1)

        from matplotlib import pyplot as plt
        for i in range(1, 2):
            plt.subplot(1, 1, i)
            plt.gray()
            plt.imshow(grids[0])
        plt.show()


if __name__ == '__main__':
    unittest.main()
