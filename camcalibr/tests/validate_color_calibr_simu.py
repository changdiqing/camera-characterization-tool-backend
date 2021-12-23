"""
To validate color calibration, we take a reference file and simulate its source file by changing the intensity,
applying difference gamma encoding to different channels and adding noise following gaussian distribution
"""
import cv2
import numpy as np
from color_calibr_ccm.color_calibr_ccm import color_calibration
from color_calibr_ccm.utils import saturate


# Add gaussian noise to an image
def noisy(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.00001
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    print(f"gauss before reshape: {gauss}")
    #gauss = gauss.reshape(row, col, ch)
    #print(f"gauss after reshape: {gauss}")
    noisy = image + gauss
    return noisy


# load the reference image
ref_color_space = "sRGB"
ref_is_linear = False
ref = [
    "imgs/color_ref_s.jpg"
]

ref_imgs = []
for img_url in ref:
    img = cv2.cvtColor(cv2.imread(img_url), cv2.COLOR_BGR2RGB)
    ref_imgs.append(img)

    # While our algorithm works with RGB, OpenCV works with BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("ref img", img)


src_color_space = "sRGB"
src_is_linear = False


# create the simulated src images from the ref images
src_imgs = []
for ref_img in ref_imgs:
    img = ref_img/255  # Normalized RGB values are suitable for modification
    # change the intensity of the image
    brightness = 0.9
    img *= brightness
    # apply different gamma encoding to the channels to simulate the color cast
    img[:, :, 0] = img[:, :, 0]**1.1
    img[:, :, 1] = img[:, :, 1]**1.1
    img[:, :, 2] = img[:, :, 2]**0.9

    # add noise to the image
    img = noisy(img)

    #  convert img to from float32 (0-1) to int8 (0-255)
    img = (img*255).astype(np.uint8)
    src_imgs.append(img)

    #  convert img to from float32 (0-1) to int8 (0-255)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("src img with noise", img)
    cv2.imwrite("imgs/color_src_simu.jpg", img)

cv2.waitKey(0)
ccm, error, calibr_imgs = color_calibration(
    src_imgs, src_color_space, src_is_linear, ref_imgs, ref_color_space, ref_is_linear, verbose=False)


for img in calibr_imgs:
    # Convert RGB to BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("calibrated:", img)
    cv2.imwrite("imgs/color_calibr_simu.jpg", img)


cv2.waitKey(0)
