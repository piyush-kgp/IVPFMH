
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str, default="images/lena.jpeg")

def histogram_equalization(img):
    w, h = img.shape # 225x225
    pdf_r = 1./(w*h)*np.bincount(img.flatten(), minlength=256)
    T = np.round(255*np.cumsum(pdf_r))
    img_eq = np.empty(img.shape, dtype=np.uint8)
    for r in range(w):
        for c in range(h):
            img_eq[r,c] = T[img[r,c]]
    Image.fromarray(img_eq.astype(np.uint8), mode='L').save("images/histogram_equalization/lena_hist_eq.jpg")

    plt.figure()
    plt.hist(img.flatten())
    plt.savefig("images/histogram_equalization/hist_orig.jpeg")
    plt.figure()
    plt.hist(img_eq.flatten())
    plt.savefig("images/histogram_equalization/hist_eq.jpeg")
    plt.figure()
    plt.plot(np.arange(256),T)
    plt.savefig("images/histogram_equalization/transform_fn.jpeg")

def main():
    args = parser.parse_args()
    img_path = args.img_path
    img = np.array(Image.open(img_path))
    img = np.mean(img, axis=2).astype(np.uint8)
    histogram_equalization(img)

if __name__=='__main__':
    main()
