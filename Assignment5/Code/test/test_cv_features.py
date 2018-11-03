import os
import unittest
import numpy as np

from Assignment5.Code import kDataPath
from PIL import Image
from utils.cv_features import divide_image, get_x_gradients, get_x_gradient_features, get_y_gradient_features, \
    get_x_gradient_histogram_features, get_y_gradient_histogram_features


class TestCVFeatures(unittest.TestCase):

    def test_get_x_gradients_image1(self):
        image = Image.open('image1.png', 'r')
        grids = divide_image(image, grid_dim=(1, 1))
        x_gradients = get_x_gradients(grids[0])
        x_gradients_w_255 = [x * 255. for x in x_gradients]
        data = np.array(x_gradients_w_255).reshape(24, 24)
        img = Image.fromarray(data)
        img.show()

    def test_get_y_gradients_image2(self):
        image = Image.open('image2.png', 'r').rotate(90)
        grids = divide_image(image, grid_dim=(1, 1))
        x_gradients = get_x_gradients(grids[0])
        x_gradients_w_255 = [abs(x * 255.) for x in x_gradients]
        data = np.array(x_gradients_w_255).reshape(24, 24)
        img = Image.fromarray(data)
        img.show()

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

        # from matplotlib import pyplot as plt
        # for i in range(1, 2):
        #     plt.subplot(1, 1, i)
        #     plt.gray()
        #     plt.imshow(grids[0])
        # plt.show()

    def test_get_x_gradient_histogram_features(self):
        fname = os.path.join(kDataPath, r'closedRightEyes/closed_eye_0001.jpg_face_1_R.jpg')
        image = Image.open(fname, 'r')
        features = get_x_gradient_histogram_features(image)
        self.assertTrue(len(features) == 5)

        fname = os.path.join(kDataPath, r'openLeftEyes/Aaron_Guiel_0001_L.jpg')
        image = Image.open(fname, 'r')
        features = get_x_gradient_histogram_features(image)
        self.assertTrue(len(features) == 5)

    def test_get_y_gradient_histogram_features(self):
        fname = os.path.join(kDataPath, r'openLeftEyes/Aaron_Guiel_0001_L.jpg')
        image = Image.open(fname, 'r')
        features = get_y_gradient_histogram_features(image)
        self.assertTrue(len(features) == 5)


if __name__ == '__main__':
    unittest.main()
