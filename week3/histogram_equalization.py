
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str, default="lena.jpeg")

def histogram_equalization(img):
    w, h = img.shape # 225x225
    memory = {}
    memory = [0 for _ in range(256)]
    for r in range(w):
        for c in range(h):
            pixel_val = img[r,c]
            memory[pixel_val] += 1
    img_eq = np.empty(img.shape, dtype=np.uint8)
    T = {}
    sum_till_now = 0
    for i, freq in enumerate(memory):
        sum_till_now += freq
        T[i] = int(255*sum_till_now/sum(memory))
    for r in range(w):
        for c in range(h):
            img_eq[r,c] = T[img[r,c]]
    Image.fromarray(img_eq.astype(np.uint8), mode='L').save("histogram_equalization.jpg")

def main():
    args = parser.parse_args()
    img_path = args.img_path
    img = np.array(Image.open(img_path))
    img = np.mean(img, axis=2).astype(np.uint8)
    Image.fromarray(img, mode='L').save("lena_bw.jpg")
    histogram_equalization(img)

if __name__=='__main__':
    main()
