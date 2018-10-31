import os
import unittest

import PIL
from Assignment5.Code import kDataPath
from PIL import Image
from utils.cv_features import divide_image


class TestCVFeatures(unittest.TestCase):

    def test_x_gradients(self):
        fname = os.path.join(kDataPath, r'closedRightEyes/closed_eye_0001.jpg_face_1_R.jpg')
        image = Image.open(fname, 'r')
        xSize = image.size[0]
        ySize = image.size[1]
        numPixels = xSize * ySize

        pixels = image.load()

        # split pixels into 3 x 3 grids

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




if __name__ == '__main__':
    unittest.main()
