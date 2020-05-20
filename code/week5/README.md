
# Optional programming exercises

- Implement the Hough transform to detect circles.
- Implement the Hough transform to detect ellipses.
- Implement the Hough transform to detect straight lines and circles in the same image.
- Consider an image with 2 objects and a total of 3 pixel values (1 for each object and one for the background). Add Gaussian noise to the image. Implement and test Otsu’s algorithm with this image.
Implement a region growing technique for image segmentation. The basic idea is to start from a set of points inside the object of interest (foreground), denoted as seeds, and recursively add neighboring pixels as long as they are in a pre-defined range of the pixel values of the seeds.
- Implement region growing from multiple seeds and with a functional like Mumford-Shah. In other words, start from multiple points (e.g., 5) randomly located in the image. Grow the regions, considering a penalty that takes into account average gray value of the region as it grows (and error it produces) as well as the new length of the region as it grows. Consider growing always from the region that is most convenient.
