
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str, default="lena.jpeg")

def add_noise(img):
    img = img.copy()
    w, h = img.shape
    noise_amt = 0.004
    salt_coords = np.random.randint(low=0, high=w, size=(int(noise_amt*w*h), 2))
    pepper_coords = np.random.randint(low=0, high=w, size=(int(noise_amt*w*h), 2))
    for x,y in salt_coords:
        img[x,y] = 255
    for x,y in pepper_coords:
        img[x,y] = 0
    Image.fromarray(img.astype(np.uint8), mode='L').save("lena_salt_and_pepper.jpg")
    return img


def median_filter(img):
    w, h = img.shape # 225x225
    img_filter_median = np.empty(img.shape, dtype=np.uint8)
    img_filter_mean = np.empty(img.shape, dtype=np.uint8)
    for r in range(1,w-1):
        for c in range(1,h-1):
            patch = img[r-1:r+2,c-1:c+2]
            median = sorted(list(patch.flatten()))[4]
            mean = int(patch.mean())
            print(r, c, patch.shape, median)
            img_filter_median[r,c] = median
            img_filter_mean[r,c] = mean
    for r in [0,w-1]:
        for c in [0,h-1]:
            img_filter_median[r,c] = img[r,c]
            img_filter_mean[r,c] = img[r,c]
    Image.fromarray(img_filter_median.astype(np.uint8), mode='L').save("median_filter.jpg")
    Image.fromarray(img_filter_mean.astype(np.uint8), mode='L').save("mean_filter.jpg")

def main():
    args = parser.parse_args()
    img_path = args.img_path
    img = np.array(Image.open(img_path))
    img = np.mean(img, axis=2).astype(np.uint8)
    Image.fromarray(img, mode='L').save("lena_bw.jpg")
    img = add_noise(img)
    median_filter(img)

if __name__=='__main__':
    main()
