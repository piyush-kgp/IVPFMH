
from PIL import Image
import numpy as np
from scipy.fftpack import dct, idct
import argparse
import huffman

parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str, default="lena.jpeg")

def dct2d(a):
    return dct(dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2d(a):
    return idct(idct(a, axis=0, norm='ortho'), axis=1, norm='ortho')

def save_img(a, fp):
    Image.fromarray((255*(a-a.min())/(a.max()-a.min())).astype(np.uint8), mode='L').save(fp)


def jpeg_compression(img):
    w, h = img.shape
    img = img.astype(np.float64)
    img_dct = np.empty(img.shape, dtype=np.float64)
    for i in range(0, w, 8):
        for j in range(0, h, 8):
            patch = img[i:i+8, j:j+8]
            img_dct[i:i+8, j:j+8] = dct2d(patch)

    threshold = 0.012
    img_dct_thresh = img_dct*(abs(img_dct) > (threshold*img_dct.max()))
    keep_perc = (img_dct_thresh!=0).sum()/(w*h)*100
    print("Compression using {}% of DCT coefficients".format(keep_perc))

    freq_map = {}
    for item in img_dct_thresh.flatten():
        if item in freq_map:
            freq_map[item] += 1
        else:
            freq_map[item] = 1
    print(freq_map.__len__(), w*h, set(img.flatten()).__len__())

    # img_rec = np.empty(img_dct_thresh.shape, dtype=np.float64)
    # for i in range(0, w, 8):
    #     for j in range(0, h, 8):
    #         patch = img_dct_thresh[i:i+8, j:j+8]
    #         img_rec[i:i+8, j:j+8] = idct2d(patch)
    # save_img(img_rec, "reconstruction_lena_dct_{}.jpg".format(keep_perc))


def main():
    args = parser.parse_args()
    img_path = args.img_path
    img = np.array(Image.open(img_path))
    img = np.mean(img, axis=2).astype(np.uint8)
    Image.fromarray(img, mode='L').save("lena_bw.jpg")
    jpeg_compression(img)

if __name__=='__main__':
    main()
