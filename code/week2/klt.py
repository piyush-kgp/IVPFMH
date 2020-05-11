
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str, default="images/lena.jpeg")

def klt(img, block_size, capture_variance):
    w, h = img.shape # 225x225
    block_array = [] # 2025x25
    for i in range(0, w, block_size):
        for j in range(0, h, block_size):
            block = img[i:i+block_size, j:j+block_size]
            block_array.append(block.flatten())
            # Image.fromarray(img[i:i+block_size, j:j+block_size], mode='L').save("crop_{}_{}.jpg".format(i, j))
    block_array = np.array(block_array).astype(np.float32)
    mu, sigma = block_array.mean(), block_array.std()
    block_array = (block_array-mu)/sigma
    cov_mat = block_array.T.dot(block_array)
    eig_val, eig_vec = np.linalg.eig(cov_mat)

    cap_var = 0
    for i, val in enumerate(eig_val):
        cap_var += val/sum(eig_val)
        if cap_var>capture_variance:
            break
    transform = eig_vec[:,:(i+1)]
    print(transform.shape)

    # compress
    data = block_array.dot(transform) # 2025x8
    print(data.shape)

    # reconstruct
    block_array_rec = data.dot(transform.T)
    print(block_array_rec.shape)
    img_rec = np.empty(img.shape)
    k = 0
    for i in range(0, w, block_size):
        for j in range(0, h, block_size):
            img_rec[i:i+block_size, j:j+block_size] = block_array_rec[k].reshape((block_size,block_size))
            k+=1
    img_rec = img_rec*sigma + mu
    Image.fromarray(img_rec.astype(np.uint8), mode='L').save("images/karhunen_loeve/karhunen_loeve_{}_{}.jpg".format(block_size, capture_variance))


def main():
    args = parser.parse_args()
    img_path = args.img_path
    img = np.array(Image.open(img_path))
    img = np.mean(img, axis=2).astype(np.uint8)
    for a in [5,9,25]:
        for b in [.7,.8,.9,.95,.97,.98,.99]:
            klt(img, a, b)



if __name__=='__main__':
    main()
