import random
from scipy.misc import imread, imsave
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
random.seed(1)


def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)


def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)


def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]


def augment_img(image_to_transform, folder, original_name):

    rot = random_rotation(image_to_transform)
    new_file_path = os.path.join(folder, 'rot_{}'.format(original_name))
    imsave(new_file_path, rot)

    noise = random_noise(image_to_transform)
    new_file_path = os.path.join(folder, 'noise_{}'.format(original_name))
    imsave(new_file_path, noise)

    hflip = random_noise(image_to_transform)
    new_file_path = os.path.join(folder, 'hflip_{}'.format(original_name))
    imsave(new_file_path, hflip)


if __name__ == "__main__":
    import os
    from Assignment8.Code import kDataPath

    closedEyeDir = os.path.join(kDataPath, "closedLeftEyes")
    for f in os.listdir(closedEyeDir):
        if f.endswith('jpg'):
            image_path = os.path.join(closedEyeDir, f)
            image_to_transform = imread(image_path)
            augment_img(image_to_transform, closedEyeDir, f)

    openEyeDir = os.path.join(kDataPath, "openLeftEyes")
    for f in os.listdir(openEyeDir):
        if f.endswith('jpg'):
            image_path = os.path.join(openEyeDir, f)
            image_to_transform = imread(image_path)
            augment_img(image_to_transform, openEyeDir, f)

    closedEyeDir = os.path.join(kDataPath, "closedRightEyes")
    for f in os.listdir(closedEyeDir):
        if f.endswith('jpg'):
            image_path = os.path.join(closedEyeDir, f)
            image_to_transform = imread(image_path)
            augment_img(image_to_transform, closedEyeDir, f)

    openEyeDir = os.path.join(kDataPath, "openRightEyes")
    for f in os.listdir(openEyeDir):
        if f.endswith('jpg'):
            image_path = os.path.join(openEyeDir, f)
            image_to_transform = imread(image_path)
            augment_img(image_to_transform, openEyeDir, f)
