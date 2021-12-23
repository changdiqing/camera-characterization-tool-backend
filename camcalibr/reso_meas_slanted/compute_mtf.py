import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from scipy import signal


def compute_mtf(image_data, verbose=False):
    # if image_data.shape[0] < 10 or image_data.shape[1] < 10:
    #    sys.exit("[Error] Image size can not be less than 10 pixels")

    # TODO: Checking the angle can be replaced by the homography of aruco tags
    min = np.amin(image_data)
    max = np.amax(image_data)

    # Detect the edge
    img_for_edge = cv2.fastNlMeansDenoising(image_data)
    edges = cv2.Canny(img_for_edge, min + 50, max - 50)
    y_edge, x_edge = np.where(edges == 255)
    z = np.polyfit(np.flipud(x_edge), y_edge, 1)
    # Z has all the variables of polynomial fitting, z[0] is the degree 1 which means the slope (tangent)
    angle_radians = np.arctan(z[0])
    angle_deg = angle_radians * (180 / np.pi)
    # Support both horizontal and vertical analysis of MTF
    # Deprecated, homography estimation of test chart photo prevents the edge from being rotated too
    # much
    # if abs(angle_deg) < 45:
    #    image_data = np.transpose(image_data)

    # Smoothing image
    # TODO: replace with hamming window
    kernel = np.ones((2, 2), np.float32) / 4
    # smooth_img = cv2.filter2D(image_data, -1, kernel)
    smooth_img = image_data  # cv2.filter2D(image_data, -1, kernel)

    sfr = get_sfr_from_image_4xoversampling(smooth_img, verbose)
    lsf = get_lsf_from_sfr(sfr)
    mtf_numeric = get_mtf_from_lsf(lsf)
    correct_curve = get_mtf_correct_curve(mtf_numeric)
    mtf_corrected = mtf_numeric / correct_curve
    x_mtf_final = np.linspace(0, 1, len(mtf_corrected))

    if verbose:
        plt.figure()
        plt.suptitle("MTF analysis", fontsize=20)
        plt.subplot(2, 3, 1)
        plt.imshow(image_data, cmap="gray")
        plt.subplot(2, 3, 2)
        plt.imshow(smooth_img, cmap="gray")
        plt.subplot(2, 3, 3)
        plt.imshow(edges, cmap="gray")
        plt.title("Detected Edge")
        plt.subplot(2, 3, 4)
        plt.title("ESF Curve")
        plt.plot(sfr, label="mean ESF")
        plt.xlabel("pixel")
        plt.ylabel("intensity")
        plt.subplot(2, 3, 5)
        plt.title("LSF Curve")
        plt.xlabel("pixel")
        plt.ylabel("DN value")
        plt.plot(lsf, "y-")
        plt.legend()
        plt.subplot(2, 3, 6)
        plt.plot(x_mtf_final, mtf_corrected, "y-")
        plt.xlabel("cycles/pixel")
        plt.ylabel("Modulation Factor")
        plt.title("MTF Curve")
        plt.legend()

        # Find MTF50
        idx = (np.abs(mtf_corrected - 0.5)).argmin()
        plt.figure()
        plt.plot(x_mtf_final, mtf_corrected, "r-", label="MTF after correction")
        plt.plot(x_mtf_final, mtf_numeric, "b--", label="MTF before correction")
        plt.plot(x_mtf_final, correct_curve, "y:", label="MTF correction factor")
        plt.plot(x_mtf_final[idx], mtf_corrected[idx], "ko", label="MTF50")
        plt.plot([x_mtf_final[idx], x_mtf_final[idx]], [mtf_corrected[idx], 0], "k:", linewidth=1)
        plt.text(x_mtf_final[idx], 0, round(x_mtf_final[idx], 4))
        plt.xlabel("cycles/pixel")
        plt.ylabel("Modulation Factor")
        plt.title("MTF Curve")
        plt.legend()
        plt.show()

        plt.show()

    return (x_mtf_final, mtf_corrected)


# New method for getting the edge spead function, using 4 times oversampling which is described in
# ISO 12233
def get_sfr_from_image_4xoversampling(data, verbose=False):

    data_diff_abs = np.absolute(diff_img_horizon(data))
    print(f"data_diff_abs: {data_diff_abs}")

    row, col = data.shape

    idxs_h = np.linspace(0, col - 1, col)
    idxs_v = np.linspace(0, row - 1, row)
    edge_x = []
    for r in data_diff_abs:
        r_weighted = r * idxs_h[:-1]
        centrol_id = r_weighted.sum() / (r.sum()) + 0.5  # +0.5 as suggested in ISO 12233
        edge_x.append(centrol_id)

    # regressive fit the centrol ids with straight line
    a, b = np.polyfit(idxs_v, edge_x, 1)

    # Init a matrix to hold the pixel's positions with respect to the edge position of the same row
    position_matrix = np.empty((row, col), dtype=float)
    for r in range(row):
        # Get the fitted edge position
        centrol_fitted = a * r + b
        # Get the position with respect to this edge position
        position_matrix[r] = idxs_h - centrol_fitted

    # shift the non negative positions by 1/4 pixel, so that the pixels in the interval [-0.25,0.25] are not catagorized
    # into one level (4x sampling means that each binning level has 0.25 pixel!).
    position_matrix[position_matrix >= 0] += 0.25

    # Compute the levels according to distance, each level has the interval of 0.25 (thus called 4 times oversampling)
    levels = (position_matrix / 0.25).astype(int)

    # Initiate the edge spread function (ESF)
    esf = np.array([])

    # Starting from the lowest level, all the way to the highest level. Compute the average value of the pixels
    # on the same level and append to esf
    for lvl in range(levels.min(), levels.max() + 1):
        # pick all the pixels on this level
        pixels = data[levels == lvl]
        # if pixels is empty. pixels.mean() return a NaN which has to be handled
        esf = np.append(esf, pixels.mean())

    # Fix the NaN values in esf (the program does not alway find samples for a 1/4 subpixel!)
    esf_fixed = fix_nan(esf)
    if verbose:
        plt.figure()
        plt.title("ESF Curve")
        plt.plot(esf, label="mean ESF")
        plt.xlabel("pixel")
        plt.ylabel("intensity")
        plt.figure()
        plt.title("ESF Curve")
        plt.plot(esf, label="mean ESF")
        plt.xlabel("pixel")
        plt.ylabel("intensity")
    return esf_fixed


def diff_img_horizon(img):
    """compute the horizontal first derivative of the rows
        method: central finite difference approximation


    Args:
        img ([numpy.array]): 2d numpy array representing an image

    return:
        img_diff ([numpy.array]): the horizontal derivative of the input img, the matrix size is maintained
    """

    # use finite difference filter with kernel [-1, 1] to get the first derivatve of each row
    img = np.asarray(img, dtype=np.float32)
    # padding the matrix in the horizontal direction, necessary for maintaining the matrix size
    img_pad = np.pad(img, ((0, 0), (1, 1)), "edge")
    print(np.diff(img))
    # get the central points
    filter = [[0.5, 0.5]]
    filtered = signal.correlate(img_pad, filter, mode="valid")
    return np.diff(img)


def get_lsf_from_sfr(sfr):

    # lsf = np.diff(sfr)
    data_pad = np.pad(sfr, (1, 1), "edge")
    filter = [0.5, 0.5]
    filtered = signal.correlate(data_pad, filter, mode="valid")
    deriv1 = np.diff(filtered)
    return deriv1


def get_mtf_from_lsf(lsf):
    mtf = np.absolute(np.fft.fft(lsf))
    mtf_nume = mtf / mtf[0]
    # the spatial frequency range is [0 4] Cycles/Pixel because of the 4xoversampling, devide by 4 to get [0 1]
    # Cycles/Pixel
    mtf_nume = mtf_nume[0 : int(len(mtf_nume) / 4)]

    return mtf_nume


# A correction factor for the numeric MTF, mainly because the Fourier transformation on discrete signal has different
# performance than the transformation on continuous signal.
# f is the spatial frequency with the unit cycles/pixel
def mtf_correction_factor(f):
    d = 1 / 4  # pixel pitch / oversampling factor
    k = 2  # k = 2 for 3-point derivative

    return max(math.sin(math.pi * d * k * f) / (math.pi * d * k * f), 0.1)


# Get the correction curve by computing a series of correction factor values for all the spatial frequency
# mtf is assumed to be on the range [0 1] Cycles/Pixel.
def get_mtf_correct_curve(mtf):
    N = len(mtf)
    x = np.linspace(0, N, N)
    correct_curve = np.array([mtf_correction_factor(xi / N) for xi in x])
    correct_curve[0] = 1
    return correct_curve


# For data array with unusable element (e.g. NaN), we fill the NaNs with interpolation
def fix_nan(data):
    nans, x = nan_helper(data)
    data[nans] = np.interp(x(nans), x(~nans), data[~nans])
    return data


# Src: https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
# An elegant way to fill teh NaNs in array using interpolation
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


if __name__ == "__main__":
    filename = "imgs/edge_8x8.png"
    image = cv2.imread(filename, 0)
    print(f"image: {image}")
    cv2.imshow("original image", image)

    compute_mtf(image, verbose=True)
