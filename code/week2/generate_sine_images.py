
import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
from numpy.fft import fft2, ifft2

def dct2d(a):
    return dct(dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2d(a):
    return idct(idct(a, axis=0, norm='ortho'), axis=1, norm='ortho')

def fft2d(a):
    return fft2(a).real

def ifft2d(a):
    return ifft2(a).real

def save_img(a, fp):
    Image.fromarray((255*(a-a.min())/(a.max()-a.min())).astype(np.uint8), mode='L').save(fp)

def sine_image(freq, img_size=512):
    x = np.sin(2*np.pi/freq*np.arange(img_size))
    X = np.stack([x]*img_size)
    # X = (255*(X-X.min())/(X.max()-X.min())).astype(np.uint8)
    X = (255*(X+1)/2).astype(np.uint8)
    im = Image.fromarray(X)
    im.save("sine_image_{}.jpg".format(freq))

    X = X.astype(np.uint64)
    X_dct = dct2d(X)
    save_img(X_dct, "dct_sine_{}.jpg".format(freq))

    X_rec = idct2d(X_dct)
    save_img(X_rec, "fft_rec_sine_{}.jpg".format(freq))

    X_fft = fft2d(X)
    save_img(X_fft, "fft_sine_{}.jpg".format(freq))

    X_rec = ifft2d(X_fft)
    save_img(X_rec, "fft_rec_sine_{}.jpg".format(freq))


def main():
    for freq in [256, 512, 1024, 2048, 4096]:
        sine_image(freq)

if __name__=="__main__":
    main()
