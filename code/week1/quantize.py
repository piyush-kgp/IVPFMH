
from PIL import Image
import numpy as np
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str, default="images/lena.jpeg")

def dequantize(img, nbits):
    img =  (img/2).astype(np.uint8)
    if nbits==1:
        img_rep = img
        img_rep[img_rep==1] = 255
    else:
        img_rep = img*(2**(8-nbits))
    Image.fromarray(img_rep, mode='L').save("images/lena_bw_{}_bits.jpg".format(nbits))
    return img

def main():
    args = parser.parse_args()
    img_path = args.img_path
    img = np.array(Image.open(img_path))
    img = np.mean(img, axis=2).astype(np.uint8)
    Image.fromarray(img, mode='L').save("images/lena_bw.jpg")
    for nbits in range(7, 0, -1):
        img = dequantize(img, nbits)

def main_opencv():
    args = parser.parse_args()
    img_path = args.img_path
    img = cv2.imread(img_path)
    img = np.mean(img, axis=2).astype(np.uint8)
    for nbits in range(7, 0, -1):
        img =  (img/2).astype(np.uint8)
        # IMWRITE_PNG_BILEVEL works only for 1 bit image
        cv2.imwrite("images/lena_bw_{}_bits_cv2.png".format(nbits), img, [cv2.IMWRITE_PNG_BILEVEL, nbits])


if __name__=='__main__':
    main()
    # main_opencv()
