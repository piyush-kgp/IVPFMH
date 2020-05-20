
from PIL import Image, ImageDraw
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str, default="images/lena.jpeg")

def add_noise(img, noise_amt=0.004):
    img = img.copy()
    w, h = img.shape
    salt_coords = np.random.randint(low=0, high=w, size=(int(noise_amt*w*h), 2))
    pepper_coords = np.random.randint(low=0, high=w, size=(int(noise_amt*w*h), 2))
    for x,y in salt_coords:
        img[x,y] = 255
    for x,y in pepper_coords:
        img[x,y] = 0
    Image.fromarray(img.astype(np.uint8), mode='L').save("images/spatial_filtering/lena_salt_and_pepper.jpg")
    return img


def spatial_filtering(img):
    w, h = img.shape # 225x225
    img_filter_median = np.empty(img.shape, dtype=np.uint8)
    img_filter_mean = np.empty(img.shape, dtype=np.uint8)
    img_guassian = np.empty(img.shape, dtype=np.uint8)
    gaussian_mask = 1./16*np.asarray([[1,2,1],[2,4,2],[1,2,1]])
    for r in range(1,w-1):
        for c in range(1,h-1):
            patch = img[r-1:r+2,c-1:c+2]
            median = sorted(list(patch.flatten()))[4]
            mean = int(patch.mean())
            img_filter_median[r,c] = median
            img_filter_mean[r,c] = mean
            img_guassian[r,c] = np.sum(patch*gaussian_mask).astype(np.uint8)
    for r in [0,w-1]:
        for c in [0,h-1]:
            img_filter_median[r,c] = img[r,c]
            img_filter_mean[r,c] = img[r,c]
            img_guassian[r,c] = img[r,c]
    Image.fromarray(img_filter_median.astype(np.uint8), mode='L').save("images/spatial_filtering/median_filter.jpg")
    Image.fromarray(img_filter_mean.astype(np.uint8), mode='L').save("images/spatial_filtering/mean_filter.jpg")
    Image.fromarray(img_guassian.astype(np.uint8), mode='L').save("images/spatial_filtering/guassian_mask.jpg")

def repititive_mean_filtering(img):
    w, h = img.shape # 225x225
    img_filter_mean = np.empty(img.shape, dtype=np.uint8)
    img_last = img
    for itr in range(100):
        for r in range(1,w-1):
            for c in range(1,h-1):
                patch = img_last[r-1:r+2,c-1:c+2]
                img_filter_mean[r,c] = int(patch.mean())
        for r in [0,w-1]:
            for c in [0,h-1]:
                img_filter_mean[r,c] = img_last[r,c]
        Image.fromarray(img_filter_mean.astype(np.uint8), mode='L').save("images/spatial_filtering/repititive_mean_filtering/mean_filter_itr_{}.jpg".format(itr+1))
        img_last = img_filter_mean
        print("Iteration", itr)
        itr+=1

def fun():
    frames = [Image.open("images/spatial_filtering/repititive_mean_filtering/mean_filter_itr_{}.jpg".format(itr+1)) for itr in range(100)]
    for i in range(100):
        draw = ImageDraw.Draw(frames[i])
        draw.text((10,10),"ITERATION {}".format(100-i))
    frames = [frames[0]]*50 + frames
    print(len(frames))
    frames[-1].save('images/spatial_filtering/lena_uncover.gif', format='GIF', append_images=frames[-2::-1], save_all=True, duration=100, loop=0)


def main():
    args = parser.parse_args()
    img_path = args.img_path
    img = np.array(Image.open(img_path))
    img = np.mean(img, axis=2).astype(np.uint8)
    img = add_noise(img)
    spatial_filtering(img)
    repititive_mean_filtering(img)

if __name__=='__main__':
    # main()
    fun()
