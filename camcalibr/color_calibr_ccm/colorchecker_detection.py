""" detects color checker in image, compute the color of the patches by averaging the color of the pixels.
    running this file along will load the imgs/input1.png

    Diqing Chang, 10.08.2021
"""
import cv2
import sys
import numpy as np
from .constants import ARUCO_DICT


def detect_colorchecker(image, row_n, col_n, aruco_type, verbose=False):
    """A function that detects the color checker and return the colors of all color patches as a 3d array. The color
    checker must be marked by 4 aruco tags at the four corners.

    Args:
        image: 3d(x, y and RGB) array, the image with color checker
        row_n: number of rows of color checker 
        col_n: number of columns of color checker
        aruco_type: type of aruco tags

    Returns:
        result: 2d array containing all RGB(or BGR) values of the color patches of the color checker  
    """
    # TODO: check if loaded images are 3D with the last dimension indicating the color

    # if colorchecker is real, row_n and col_n can not be less than 1
    if row_n < 1 or col_n < 1:
        print("[ERROR] Invalid row number or column number.")
        sys.exit(0)

    # Load the input image from disk and resize it
    print("[INFO] loading image...")
    # TODO: check this resize
    #image = cv2.imread(filename)
    #image = imutils.resize(image, width=600)
    # verify that the supplied ArUCo tag exists and is supported by OpenCV
    if ARUCO_DICT.get(aruco_type, None) is None:
        print("[ERROR] ArUCo tag of '{}' is not supported".format(aruco_type))
        sys.exit(0)

    # load the ArUCo dictionary, grab the ArUCo parameters, and detect
    # the markers
    print("[INFO] detecting '{}' tags...".format(aruco_type))
    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), arucoDict,
                                                       parameters=arucoParams)

    # verify that exactly 4 ArUco markers were detected
    # flatten the ArUco IDs list
    ids = ids.flatten()
    print(f"ids: {ids}")

    if not all(x in ids for x in [1, 2, 3, 4]):
        print("[ERROR] Not all four aruco tags with id 1, 2, 3 and 4 are detected")
        sys.exit(0)

    cc_topleft = None
    cc_topright = None
    cc_bottomleft = None
    cc_bottomright = None
    # loop over the detected ArUCo corners
    for (marker_corner, marker_id) in zip(corners, ids):
        # extract the marker corners (which are always returned in
        # top-left, top-right, bottom-right, and bottom-left order)
        corners = marker_corner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners
        # The coordinates are doubles, because the algorithm cann't tell you which pixel is the corner, but rather the
        # exact position. We convert the position to integer to know which pixel to select.
        if marker_id == 1:
            cc_topleft = np.array(bottomRight)
        elif marker_id == 2:
            cc_topright = np.array(bottomLeft)
        elif marker_id == 3:
            cc_bottomright = np.array(topLeft)
        else:
            cc_bottomleft = np.array(topRight)
        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))

        if verbose:
            # draw the bounding box of the ArUCo detection
            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

            # draw the ArUco marker ID on the image
            cv2.putText(image, str(marker_id),
                        (topLeft[0], topLeft[1] -
                        15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
        print("[INFO] ArUco marker ID: {}".format(marker_id))

    # Keystone correction of image according to detected aruco tags
    cc_topleft_corrected = cc_topleft
    cc_topright_corrected = np.array([cc_topright[0], cc_topleft[1]])
    cc_bottomleft_corrected = np.array([cc_topleft[0], cc_bottomleft[1]])
    cc_bottomright_corrected = np.array([cc_topright[0], cc_bottomleft[1]])
    # compute padding for each color patch, currently we use 25%
    padding = 0.25
    padding_h = (cc_topright_corrected[0] -
                 cc_topleft_corrected[0])/col_n*padding
    padding_v = (cc_bottomleft_corrected[1] -
                 cc_topleft_corrected[1])/row_n*padding
    H = cv2.findHomography(np.array([cc_topleft, cc_topright, cc_bottomright, cc_bottomleft]), np.array([
                           cc_topleft_corrected, cc_topright_corrected, cc_bottomright_corrected, cc_bottomleft_corrected]), cv2.LMEDS)
    dstD = cv2.warpPerspective(image, H[0], (image.shape[1], image.shape[0]))

    # create the grid coordinates according to the keystone corrected corners
    x, y = np.mgrid[cc_topleft_corrected[0]:cc_topright_corrected[0]:(
        col_n+1) * 1j, cc_topleft_corrected[1]: cc_bottomleft_corrected[1]:(row_n+1) * 1j]
    x = x.T
    y = y.T

    # The result should be matrix holding the RGB value of all col_n*row_n color patches, init with zeros
    result = np.zeros((row_n * col_n, 3), dtype=int)
    for i in range(row_n):
        for j in range(col_n):
            # draw the bounding box of each color patch
            topLeft = (int(x[i][j] + padding_h), int(y[i][j] + padding_v))
            topRight = (int(x[i][j+1] - padding_h), int(y[i][j+1] + padding_v))
            bottomRight = (int(x[i+1][j+1] - padding_h),
                           int(y[i+1][j+1] - padding_v))
            bottomLeft = (int(x[i+1][j]+padding_h), int(y[i+1][j] - padding_v))

            # Feed result with the average RGB of the central area (color patch inside the 25% padding)
            rgb_avg = dstD[topLeft[1]:bottomLeft[1],
                           topLeft[0]: topRight[0]].mean(axis=(0, 1))
            result[i*col_n + j] = rgb_avg  # [::-1]

            # DEBUG ony: mark the area used for picking color
            if verbose:
                cv2.line(dstD, topLeft, topRight, (0, 255, 0), 1)
                cv2.line(dstD, topRight, bottomRight, (0, 255, 0), 1)
                cv2.line(dstD, bottomRight, bottomLeft, (0, 255, 0), 1)
                cv2.line(dstD, bottomLeft, topLeft, (0, 255, 0), 1)

    # show the output image
    if verbose:
        cv2.imshow("Original Image", image)
        cv2.imshow("keystone corrected with grid", dstD)
        cv2.imwrite("imgs/output_cc_detected.png", dstD)
        cv2.waitKey(0)
    return result


if __name__ == '__main__':
    filename = "imgs/input1_aruco_ref.png"
    detect_colorchecker(filename, 4, 6, "DICT_5X5_50", verbose=True)
