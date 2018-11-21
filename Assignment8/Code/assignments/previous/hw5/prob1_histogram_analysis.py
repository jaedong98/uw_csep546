import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from Assignment5.Code import kDataPath, report_path
from utils.cv_features import get_x_gradients, divide_image

x_gradents = []
y_gradents = []
for f in os.listdir(kDataPath):
    if os.path.isfile(f):
        continue
    folder = os.path.join(kDataPath, f)
    for img_f in os.listdir(folder):
        if not img_f.endswith('jpg'):
            continue
        ipath = os.path.join(folder, img_f)
        image = Image.open(ipath, 'r')
        grid = divide_image(image, grid_dim=(1, 1))[0]
        x_g = get_x_gradients(grid)
        x_gradents.extend([abs(x) for x in x_g])

        image = Image.open(ipath, 'r').rotate(90)
        grid = divide_image(image, grid_dim=(1, 1))[0]
        y_g = get_x_gradients(grid)
        y_gradents.extend([abs(x) for x in y_g])

print("Max in x:{}".format(max(x_gradents)))
print("Max in y:{}".format(max(y_gradents)))

n_bins = 50

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

axs[0].set_title("X histogram gradient counts")
axs[1].set_title("Y histogram gradient counts")

# We can set the number of bins with the `bins` kwarg
axs[0].hist(x_gradents, bins=n_bins, facecolor='green', alpha=0.75)
axs[1].hist(y_gradents, bins=n_bins, facecolor='green', alpha=0.75)
plt.show()

img_fname = os.path.join(report_path, "prob1_histogram_analaysis.png")
fig.savefig(img_fname)


