
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str, default="images/lena.jpeg")

def scaling(img, block_size):
    n_rows, n_cols = img.shape
    img_new = np.empty((n_rows//block_size, n_cols//block_size), dtype=np.uint8)
    # print(img_new.shape, n_rows, n_cols)
    for row in range(0, n_rows, block_size):
        for col in range(0, n_cols, block_size):
            patch = img[row:row+block_size, col:col+block_size]
            if patch.shape != (block_size, block_size):
                continue
            # print(row, col, patch.shape)
            img_new[row//block_size, col//block_size] = np.mean(patch).astype(np.uint8)
    Image.fromarray(img_new, mode='L').save("images/lena_scaled_{}.jpg".format(block_size))


def main():
    args = parser.parse_args()
    img_path = args.img_path
    img = np.array(Image.open(img_path))
    img = np.mean(img, axis=2).astype(np.uint8)
    for block_size in [3, 5, 7]:
        scaling(img, block_size)

if __name__=='__main__':
    main()
