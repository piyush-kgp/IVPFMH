
from PIL import Image
import numpy as np
import argparse
import matplotlib.pyplot as plt
import heapq


parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str, default="lena.jpeg")


class Node:
    def __init__(self, p, s, l, r):
        self.data = p
        self.sign = s
        self.left = l
        self.right = r
    def __le__(self, other):
        return self.data<=other.data
    def __lt__(self, other):
        return self.data<other.data

def encode(root, prefix, codes):
    if root is None:
        return
    if root.left is None and root.right is None:
        codes[root.sign] = prefix
    p1 = prefix+'0'
    p2 = prefix+'1'
    encode(root.left, p1, codes)
    encode(root.right, p2, codes)

def huffman(freq_map):
    pq = []
    for s, p in freq_map.items():
        n = Node(p,s,None,None)
        heapq.heappush(pq, n)
    while len(pq)>1:
        left = heapq.heappop(pq)
        right = heapq.heappop(pq)
        root = Node(left.data+right.data,'\0', left, right)
        heapq.heappush(pq, root)
    codes = {}
    encode(root, "", codes)
    return codes

def test_huffman_encoding():
    freq_map = {'A': 0.1, 'B': 0.4, 'C': 0.06, 'D': 0.1, 'E': 0.04, 'F': 0.3}
    huffman_codes = huffman(freq_map)
    print(huffman_codes)


def predictive_lossless_compression(img):
    # compression
    img = img.astype(np.float32)
    error_table = np.empty(img.shape, dtype=np.float32)
    w, h = img.shape
    for i in range(w):
        for j in range(h):
            if i==j==0:
                error_table[i,j] = img[i,j]
            elif i==0:
                error_table[i,j] = img[i,j] - img[i,j-1]
            elif j==0:
                error_table[i,j] = img[i,j] - img[i-1,j]
            else:
                error_table[i,j] = img[i,j] - (img[i-1,j-1]+img[i-1,j]+img[i,j-1])/3
    plt.hist(error_table.flatten(), bins=100)
    plt.savefig("error_histogram.jpg")
    # After this we need the Huffman Codes for the Error based on above histogram.
    freq_map = {}
    error_table = error_table.astype(np.int64)
    for num in error_table.flatten():
        if num not in freq_map:
            freq_map[num] = 1
        else:
            freq_map[num] += 1
    huffman_codes = huffman(freq_map)
    # print(freq_map, huffman_codes)
    # print(len(huffman_codes)) # 237

    # huffman_codes = {k: int(v, 2) for k, v in huffman_codes.items()}
    f = open("compressed_image.txt", 'a+')
    for r in error_table:
        for c in r:
            f.write(str(int(huffman_codes[c], 2)))
            f.write(',')
        f.write('\n')
    f.close()
    return huffman_codes, "compressed_image.txt", img.shape


def reconstruct(huffman_codes, shape, filename):
    w,h = shape
    error_table = [line.split(',')[:-1] for line in open("compressed_image.txt", 'r').read().split('\n')[:-1]]
    error_table = np.array(error_table).astype(np.int64)
    huffman_codes_rev = {int(v, 2): k for k, v in huffman_codes.items()}

    for i in range(w):
        for j in range(h):
            error_table[i,j] = huffman_codes_rev[error_table[i,j]]

    # print(error_table)

    # reconstruction
    img_rec = np.empty(shape, dtype=np.float32)
    for i in range(w):
        for j in range(h):
            if i==j==0:
                img_rec[i,j] = error_table[i,j]
            elif i==0:
                img_rec[i,j] = img_rec[i,j-1] + error_table[i,j]
            elif j==0:
                img_rec[i,j] = img_rec[i-1,j] + error_table[i,j]
            else:
                img_rec[i,j] = (img_rec[i-1,j-1]+img_rec[i-1,j]+img_rec[i,j-1])/3 + error_table[i,j]
    # print(img_rec)
    Image.fromarray(img_rec.astype(np.uint8), mode='L').save("lossless_compression.jpg")



def main():
    args = parser.parse_args()
    img_path = args.img_path
    img = np.array(Image.open(img_path))
    img = np.mean(img, axis=2).astype(np.uint8)
    Image.fromarray(img, mode='L').save("lena_bw.jpg")
    huffman_codes, filename, shape =predictive_lossless_compression(img)
    reconstruct(huffman_codes, shape, filename)
    # test_huffman_encoding()

if __name__=='__main__':
    main()
