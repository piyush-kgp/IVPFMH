
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str)

def rotate(img, theta):
    theta_ = np.pi/180*theta
    T = np.array([[np.cos(theta_), np.sin(theta_)], [-np.sin(theta_), np.cos(theta_)]])
    n_rows, n_cols, _ = img.shape
    img_transformed = np.empty((n_rows*2, n_cols*2, 3), dtype=np.uint8)
    for row in range(n_rows):
        for col in range(n_cols):
            pixel_data = img[row, col]
            coords = np.array([row, col])
            row_new, col_new = T.dot(coords).astype(np.int)
            img_transformed[row_new, col_new] = pixel_data
    Image.fromarray(img_transformed, mode='RGB').save("lena_rotated_{}.jpg".format(theta))



def main():
    args = parser.parse_args()
    img_path = args.img_path
    img = np.array(Image.open(img_path))
    # img = np.mean(img, axis=2).astype(np.uint8)
    for theta in [90, 45]:
        rotate(img, theta)

if __name__=='__main__':
    main()
