
def get_integrity_array(image, width=8):

    for x in range(3):
        for y in range(3):
            print(x, y, (y * width) + x)



def get_x_gradient(image):
    pass

def get_y_gradient(image):
    image = image.rotate(90)  # rotate 90 in counter clockwise
    pass

def get_min_max_avg_gradient(image):
    pass


get_integrity_array(None)