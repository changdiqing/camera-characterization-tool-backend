"""
For resolution (MTF), we need to generate a simulated slanted edge, whose analytical MTF is known, then execute the
measurements and compare the results 
"""
import numpy as np
import matplotlib.pyplot as plt
from camcalibr.reso_meas_slanted.compute_mtf import (
    get_sfr_from_image_4xoversampling,
    compute_mtf,
    get_lsf_from_sfr,
    get_mtf_from_lsf,
)


def sigmoid(x, gamma, phi):
    # phase shift to let the theoretical curve match the measured curve
    x -= phi
    return 1 - 0.5 * np.exp(-gamma * x) if x >= 0 else 0.5 * np.exp(gamma * x)


def lsf(x, gamma):
    sign = -1 if x >= 0 else 1
    return 0.5 * gamma * np.e ** (sign * gamma * x)


def mtf(epsilon, gamma):
    return (gamma ** 2) / (gamma ** 2 + (2 * np.pi * epsilon) ** 2)


def draw_edge(angle, row, col, gamma):
    """draw an edge with the given angle agains the vertical line, each row has the edge spread function that is exact
            a sigmoid function

    Args:
            angle: unit degree, the angle between the slanted edge and the vertical line
            row and col: size of the numpy.array
    """

    result = np.zeros((row, col))
    row_mid = (row - 1) / 2.0
    col_mid = (col - 1) / 2.0
    angle_radiant = np.pi * angle / 180
    x = np.linspace(0, col - 1, col) - col_mid

    for i in range(row):
        phaseshift = (i - row_mid) * np.tan(angle_radiant)
        result[i] = np.array([round(sigmoid(xi, gamma, phaseshift) * 255) for xi in x])

    return (result).astype(np.uint8)


def xnorm(data1d):
    return np.linspace(0, 1, len(data1d))


def mse(src, ref):
    return ((src - ref) ** 2).mean()


angle = 5
row = 100  # 256
col = 100  # 201

gamma = 1.8
row_mid = row / 2
col_mid = (col - 1) / 2
edge_img = draw_edge(angle, row, col, gamma)
x = np.linspace(0, col - 1, col) - col_mid

# edge_img = cv2.imread('imgs/synthetic_slanted_edge.png', cv2.IMREAD_GRAYSCALE)


# esf
measesf = get_sfr_from_image_4xoversampling(edge_img)
xdata = np.linspace(0, (len(measesf) - 1), len(measesf))


# number of columns of the oversamped ESF
col_os = len(measesf)
col_mid_os = col_os / 2


# Chop off the ends to ensure that the oversampled ESF has the same spatial frequency range as each row of the image
chophead = int((col_os - col * 4) / 2)
choptail = col_os - col * 4 - chophead
measesf_chopped = measesf[chophead:-choptail] if choptail != 0 else measesf[chophead:]
# oversampling will have 3 pixels out of the spatial frequency range of each row, therefore has to be removed
# Another explaination: The generated ESF after 4xoversampling is like inserting 3 additional pixels BETWEEN the two
# adjacent original pixels, so the oversampled signal of a n pixel signal should have 4n-3 pixels
measesf_chopped = measesf_chopped[1:-2]
# Normalize the chopped measured ESF for comparing with the theoretical ESF
measesf_chopped /= np.float32(255)


# theoesf = np.array([sigmoid(xi, gamma, 0) for xi in x])
theoesf = np.array([sigmoid(xi, gamma, 0) for xi in x])

# lsf
theolsf = np.array([lsf(xi, gamma) for xi in x])
measlsf_chopped = get_lsf_from_sfr(measesf_chopped)
measlsf = get_lsf_from_sfr(measesf)


# mtf
x_mtf, y_mtf = compute_mtf(edge_img)
theomtf = np.array([mtf(xi, gamma) for xi in x_mtf])
mtf_theolsf_full = np.absolute(np.fft.fft(theolsf))
mtf_theolsf = mtf_theolsf_full / mtf_theolsf_full[0]
mtf_theolsf = mtf_theolsf[0 : int(len(mtf_theolsf))]

mtf_measlsf = np.absolute(np.fft.fft(measlsf_chopped))
mtf_measlsf = mtf_measlsf / mtf_measlsf[0]
mtf_measlsf = mtf_measlsf[0 : int(len(mtf_measlsf) / 4)]

plt.figure()
plt.imshow(edge_img, cmap="gray")
plt.figure()
plt.xlabel("Pixel")
plt.ylabel("Normalized ESF")
print(f"ERROR of ESF (RMSE):{np.sqrt(mse(theoesf, measesf_chopped[0::4]))} ")
plt.plot(theoesf, "b--", label="Theoretical ESF", linewidth=2)

plt.plot(
    np.linspace(0, col - 1, len(measesf_chopped)),
    measesf_chopped,
    "g-",
    label="Measured ESF",
    linewidth=2,
    marker="o",
    alpha=0.3,
)
plt.legend()
# LSF
plt.figure()
plt.xlabel("Normalized Pixel Position")
plt.ylabel("Normalized LSF")
plt.plot(x, theolsf, "b--", label="Theoretical LSF", linewidth=2)
plt.plot(xnorm(measlsf_chopped), measlsf_chopped * 4 / 255, "k-", label="Meas LSF From Meas ESF", linewidth=2)
plt.legend()
plt.figure()
plt.xlabel("Spatial Frequency (Cycles/Pixel)")
plt.ylabel("Normalized MTF")
plt.plot(x_mtf, theomtf, "b-", label="Theoretical MFT", linewidth=2),
plt.plot(x_mtf, y_mtf, "k-", marker="^", label="Measured MFT", linewidth=2, alpha=0.3)
print(f"Error of MTF (RMSE):{np.sqrt(mse(theomtf, y_mtf))}")

plt.legend()


plt.show()
