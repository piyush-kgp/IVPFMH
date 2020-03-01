
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str)

def spatial_averaging(img, n_neighbors):
    b = int(n_neighbors/2)
    img = np.pad(img, pad_width=b, mode="symmetric")
    n_rows, n_cols = img.shape
    img_new = np.empty(img.shape, dtype=np.uint8)
    for row in range(b,n_rows-b):
        for col in range(b,n_cols-b):
            patch = img[row-b:row+b+1,col-b:col+b+1]
            img_new[row,col] = np.mean(patch).astype(np.uint8)
    img_new = img_new[b:-b,b:-b]
    Image.fromarray(img_new, mode='L').save("lena_spatially_averaged_{}_pixels.jpg".format(n_neighbors))


def main():
    args = parser.parse_args()
    img_path = args.img_path
    img = np.array(Image.open(img_path))
    img = np.mean(img, axis=2).astype(np.uint8)
    for n_neighbors in [3,10,20]:
        spatial_averaging(img, n_neighbors)

if __name__=='__main__':
    main()
