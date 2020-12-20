import numpy as np
import scipy
import scipy.signal as sig
from imageio import imread
import matplotlib.pyplot as plt


def normalize(m, min_v, max_v):
	return (m - min_v) / (max_v - min_v) * 255


def norm(img):
	return normalize(img, np.min(img), np.max(img)).astype(int)


def rgb2gray(rgb):
	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
	gray = norm(0.2989 * r + 0.5870 * g + 0.1140 * b)
	return gray


# With mode="L", we force the image to be parsed in the grayscale, so it is
# actually unnecessary to convert the photo color beforehand.
img = imread("~/Downloads/manu-2004.jpg")
img = np.array(img)
img = rgb2gray(img)

# Define the Sobel operator kernels.
kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

G_x = norm(sig.convolve2d(img, kernel_x, mode='same'))
G_y = norm(sig.convolve2d(img, kernel_y, mode='same'))


# Plot them!
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# Actually plt.imshow() can handle the value scale well even if I don't do 
# the transformation (G_x + 255) / 2.
ax1.imshow(G_x , cmap='gray'); ax1.set_xlabel("Gx")
ax2.imshow(G_y , cmap='gray'); ax2.set_xlabel("Gy")
plt.show()

