
import heapq

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
    encode(root.left, prefix+"0", codes)
    encode(root.right, prefix+"1", codes)

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

def test():
    freq_map = {'A': 0.1, 'B': 0.4, 'C': 0.06, 'D': 0.1, 'E': 0.04, 'F': 0.3}
    huffman_codes = huffman(freq_map)
    print(huffman_codes)

if __name__=="__main__":
    main()
