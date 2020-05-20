
# Optional programming exercises

- Implement the basic non-local means type of inpainting approach. Do this both for still and for video, where the search for similar patches in the latter is across multiple frames.
- For a given image, compute at every pixel the inner product between the gradient of the Laplacian and the level lines normal, this being the main term in one of the inpainting techniques we learned. Display it and analyze its behavior.
- For a given video, implement a very simple inpainting technique: At every pixel to be filled-in, inpaint it with the median of the values (if available) for pixels at the same spatial position corresponding to N frames before and N frames after, varying the value of N. Extend this to consider camera motion, in particular by exploiting registration techniques as available in Matlab and other packages.
