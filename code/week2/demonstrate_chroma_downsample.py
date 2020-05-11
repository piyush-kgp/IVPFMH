
"""
The purpose of this file is to show that you can get away with large (say 10x in both directions)
downsample in the chroma component without much change of perception to human eye.
"""

from PIL import Image
import numpy as np

def downsample_cb_cr(ycbcr, ds_level):
    w, h = ycbcr.size
    ycbcr_np = np.asarray(ycbcr)
    ycbcr_res = ycbcr.resize((w//ds_level, h//ds_level))
    ycbcr_ds = ycbcr_res.resize((w, h))
    ycbcr_ds_np = np.array(ycbcr_ds)
    ycbcr_ds_np[:,:,0] = ycbcr_np[:,:,0] # Cb, Cr channels are downsampled, Y is kept as it is
    return Image.fromarray(ycbcr_ds_np, mode='YCbCr')

def main():
    img = Image.open("images/lena.jpeg")
    img_np = np.array(img)
    ycbcr = img.convert('YCbCr')
    for level in [2,5,10,20,30,40,50,80,100,200]:
        ycbcr_new = downsample_cb_cr(ycbcr, level)
        ycbcr_new.save("images/chroma_ds/lena_chroma_ds_level_{}.jpeg".format(level))

if __name__=='__main__':
    main()
