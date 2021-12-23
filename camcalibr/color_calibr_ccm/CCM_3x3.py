from scipy.optimize import minimize, fmin
from cv2 import cv2
import sys
import numpy as np
from .linearize import *
from .colorspace import *
from .colorchecker import *
from .utils import *
from .distance import *

# TODO: add support for weights
# TODO: input src and dst should be already converted to the correct color space!!!


class CCM_3x3:
    def __init__(
        self,
        src,
        dst,
        dst_colorspace="sRGB",
        dst_illuminant="D65",
        dst_observer="2",
        saturated_threshold=(0.02, 0.98),
        colorspace="sRGB",
        linear="gamma",
        gamma=2.2,
        deg=3,
        distance="de00",
        dist_illuminant="D65",
        dist_observer="2",
        xtol=1e-4,
        ftol=1e-4,
    ):

        # src
        self.src = src / 255.0
        self.dst = dst / 255.0

        # instantiate color spaces for reference image and target image
        dist_io = IO(dist_illuminant, dist_observer)
        self.src_cs = globals()[colorspace]()
        self.src_cs.set_default(dist_io)
        dst_io = IO(dst_illuminant, dst_observer)
        self.dst_cs = globals()[dst_colorspace]()
        self.dst_cs.set_default(dst_io)

        # linear method
        # TODO:
        # self.linear = globals()['Linear_'+linear](gamma,
        #                                         deg, src, self.cc, saturated_threshold)

        # TODO:
        saturate_mask = saturate(self.src, *saturated_threshold)
        self.mask = saturate_mask
        # TODO: the gamma function works here better than the linearization function of sRGB on wiki
        # self.src_rgbl = self.src_cs.rgb2rgbl(self.src)
        self.src_rgbl = Linear_gamma(gamma).linearize(self.src)
        self.src_rgbl_masked = self.src_rgbl[self.mask]

        # TODO: reference color space should be limited to only RGB!!
        # self.dst_cs.lab2rgbl(self.dst)[self.mask]
        self.dst_rgb_masked = self.dst[self.mask]
        self.dst_rgbl_masked = Linear_gamma(gamma).linearize(self.dst_rgb_masked)
        self.dst_lab_masked = self.dst_cs.rgbl2lab(self.dst_rgbl_masked)

        self.masked_len = len(self.src_rgbl_masked)
        if self.masked_len == 0:
            print("[ERROR] image has the over saturated RGB values.")
            sys.exit(0)

        # The nonlinear optimization options:
        # 1. distance function
        self.distance = globals()["distance_" + distance]

        # 2. xtol and ftol
        self.xtol = xtol
        self.ftol = ftol
        # the output
        self.ccm = None

        # distance function may affect the loss function and the calculate function
        # 'rgbl distance'
        if distance == "rgb":
            self.calculate_rgb()
        elif distance == "rgbl":
            self.calculate_rgbl()
        else:
            self.calculate()

    # TODO:
    def initial_white_balance(self, src_rgbl, dst_rgbl):
        """calculate nonlinear-optimization initial value by white balance:
        res = diag(mean(s_r)/mean(d_r), mean(s_g)/ \
                   mean(d_g), mean(s_b)/mean(d_b))
        https://www.imatest.com/docs/colormatrix/
        """
        rs, gs, bs = np.sum(src_rgbl, axis=0)
        rd, gd, bd = np.sum(dst_rgbl, axis=0)
        return np.array([[rd / rs, 0, 0], [0, gd / gs, 0], [0, 0, bd / bs]])

    def initial_least_square(self, src_rgbl, dst_rgbl):
        """calculate nonlinear-optimization initial value by least square:
        res = np.linalg.lstsq(src_rgbl, dst_rgbl)
        """
        return np.linalg.lstsq(src_rgbl, dst_rgbl, rcond=None)[0]

    def loss_rgb(self, ccm):
        """loss function if distance function is rgb
        it is square-sum of color difference between src_rgbl@ccm and dst
        """
        ccm = ccm.reshape((-1, 3))
        rgb_est = self.src_cs.rgbl2rgb(self.src_rgbl_masked @ ccm)
        dist = self.distance(rgb_est, self.dst_rgb_masked)
        dist = np.power(dist, 2)
        return sum(dist)

    def calculate_rgb(self):
        """calculate ccm if distance function is rgb"""
        ccm0 = self.inital_func(self.src_rgbl_masked, self.dst_rgbl_masked)
        ccm0 = ccm0.reshape((-1))
        res = fmin(self.loss_rgb, ccm0, xtol=self.xtol, ftol=self.ftol)
        if res is not None:
            self.ccm = res.reshape((-1, 3))
            self.error = (self.loss_rgb(res) / self.masked_len) ** 0.5

    def loss_rgbl(self, ccm):
        dist = np.sum(np.power(self.dst_rgbl_masked - self.src_rgbl_masked @ self.ccm, 2), axis=-1)
        return sum(dist)

    def calculate_rgbl(self):
        self.ccm = self.ccm_by_least_square(self.src_rgbl_masked, self.dst_rgbl_masked)
        self.error = (self.loss_rgbl(self.ccm) / self.masked_len) ** 0.5

    def loss(self, ccm):
        """
        loss function of de76 de94 and de00
        it is square-sum of color difference between src_rgbl@ccm and dst
        """
        ccm = ccm.reshape((-1, 3))
        lab_est = self.src_cs.rgbl2lab(self.src_rgbl_masked @ ccm)
        dist = self.distance(lab_est, self.dst_lab_masked)
        dist = np.power(dist, 2)
        return sum(dist)

    def calculate(self):
        """calculate ccm if distance function is de76 de94 and de00"""
        ccm0 = self.initial_least_square(self.src_rgbl_masked, self.dst_rgbl_masked)
        ccm0 = ccm0.reshape((-1))
        res = fmin(self.loss, ccm0, xtol=self.xtol, ftol=self.ftol)
        if res is not None:
            self.ccm = res.reshape((-1, 3))
            self.error = (self.loss(res) / self.masked_len) ** 0.5

    def value(self):

        # over-saturation rate
        # We generate 10000 pixels with random RGB values in the range [0,1]
        n_pixels = 1000
        img_rand = np.random.randint(256, size=(n_pixels, 3)) / 255

        mask = saturate(self.infer(img_rand), 0, 1)
        self.sat = np.sum(mask) / n_pixels
        print("sat:", self.sat)
        # TODO: to be fixed, does not work yet. Distribution
        # rgbl = self.src_cs.rgb2rgbl(rand)
        # mask = saturate(rgbl@np.linalg.inv(self.ccm), 0, 1)
        # self.dist = np.sum(mask)/number
        # print('dist:', self.dist)

        return self.ccm, self.error

    def infer(self, img):
        """infer using calculated ccm"""
        if self.ccm is None:
            raise Exception("No CCM values!")
        img_lin = self.src_cs.rgb2rgbl(img)
        img_ccm = img_lin @ self.ccm

        infered_img = self.src_cs.rgbl2rgb(img_ccm)

        return infered_img

    def infer_image(self, img, in_size=255, out_size=255, out_dtype=np.uint8):
        """infer image and output as an BGR image with uint8 type"""

        img_norm = img / in_size
        out = self.infer(img_norm)

        # Avoid color blow out
        saturate_mask = saturate(out, 0, 1, axis=2)  # find out the valid pixels

        blowout_mask = np.invert(saturate_mask)  # find out the blow out pixels

        out[blowout_mask] = img_norm[blowout_mask]  # replace the blow out pixels with the originals

        out = np.minimum(np.maximum(np.round(out * out_size), 0), out_size)  # convert range 0-1 to 0-255

        out = out.astype(out_dtype)
        return out
