"""
Test the color calibration
With scatter3d set to True, the program draws the 3d scatter plots of the colors of the calibrated images. One of the 
calibrated images is simply a matrix of random RGB values. The 3d scatter plotting visualizes whether the colors stay in
the safe space (the [0 1]^3 space) or blow out (out of the [0 1]^3 space)
"""
from color_calibr_ccm.color_calibr_ccm import color_calibration
import matplotlib.pyplot as plt
from color_calibr_ccm.utils import *
from itertools import product, combinations
import cv2
import numpy as np

# Draw the 3D scatter of the colors to visualize the color blow-out (out side of the [0 1]^3 space)
scatter3d = False

src = ["imgs/color_src_s.jpg"]
src_color_space = "sRGB"
src_is_linear = False

ref = ["imgs/color_ref_s.jpg"]
ref_color_space = "sRGB"
ref_is_linear = False

src_imgs = []
for img_url in src:
    img = cv2.cvtColor(cv2.imread(img_url), cv2.COLOR_BGR2RGB)
    src_imgs.append(img)

ref_imgs = []
for img_url in ref:
    img = cv2.cvtColor(cv2.imread(img_url), cv2.COLOR_BGR2RGB)
    ref_imgs.append(img)

# Set distance to test calibration with loss:
#  distance = "rgb": loss is the MSE of RGB values
#  distance = "rgbl": loss is the MSE of linear RGB values
#  distance = "de00": loss is the CIEDE2000 error (in CIELAB color space)
distance = "de00"
ccm, error, calibr_imgs = color_calibration(
    src_imgs, src_color_space, src_is_linear, ref_imgs, ref_color_space, ref_is_linear, distance=distance, verbose=False
)

print(f"ccm: {ccm}")
print(f"error: {error}")

for img in calibr_imgs:
    # Convert RGB to BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("imgs/color_calibr_" + distance + ".jpg", img)


if scatter3d:
    # Generate a image with random pixels and check if it oversaturates
    n_pixels = 1000
    img_rand = np.random.randint(256, size=(n_pixels, 3)) / 255

    def scatterRGB3D(img, label):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.set_xlabel("Red")
        ax.set_ylabel("Green")
        ax.set_zlabel("Yellow")

        for rgb in img:
            if np.all(np.logical_and(rgb >= 0, rgb <= 1)):
                color = (rgb[0], rgb[1], rgb[2])
            else:
                color = (0, 0, 0)
            ax.scatter3D(rgb[0], rgb[1], rgb[2], marker="o", color=color)

        # draw cube
        r = [0, 1]
        for s, e in combinations(np.array(list(product(r, r, r))), 2):
            if np.sum(np.abs(s - e)) == r[1] - r[0]:
                ax.plot3D(*zip(s, e), color="gray")

    src_img = src_imgs[0]
    ref_img = ref_imgs[0]
    calibr_img = calibr_imgs[0]

    scatterRGB3D(calibr_img / 255, "calibr")
    scatterRGB3D(img_rand, "img_rand")

plt.show()
