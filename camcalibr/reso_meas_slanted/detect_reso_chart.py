import cv2
import sys
import numpy as np


def detect_reso_chart(image, verbose=False):
    """ Detect the resolution test chart from image (numpy.array, one rotated rectangular with 4 slanted edges)
        if photo is valid, (clear enough to recognise the four Aruco tags and camera is facing the target directly:
            extract the 4 slanted  edges
        else:
            return None

    Args:
        image ([numpy.array]): Photo of resolution test chart (for example see test_chart/reso_test_chart.png))
    """
    # Aruco tag has to be Dict_
    aruco_type = cv2.aruco.DICT_5X5_50

    # Recognise the 4 aruco tags
    print("[INFO] detecting '{}' tags...".format(aruco_type))
    arucoDict = cv2.aruco.Dictionary_get(aruco_type)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,
                                                       parameters=arucoParams)
    # verify that exactly 4 ArUco markers were detected
    # flatten the ArUco IDs list
    ids = ids.flatten()
    print(f"ids: {ids}")

    if not all(x in ids for x in [1, 2, 3, 4]):
        sys.exit(
            "[ERROR] Not all four aruco tags with id 1, 2, 3 and 4 are detected")

    rc_topleft = None
    rc_topright = None
    rc_bottomleft = None
    rc_bottomright = None
    # loop over the detected ArUCo corners
    for (marker_corner, marker_id) in zip(corners, ids):
        # extract the marker corners (which are always returned in
        # top-left, top-right, bottom-right, and bottom-left order)
        corners = marker_corner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners
        # The coordinates are doubles, because the algorithm cann't tell you which pixel is the corner, but rather the
        # exact position. We convert the position to integer to know which pixel to select.
        if marker_id == 1:
            rc_topleft = np.array(bottomRight)
        elif marker_id == 2:
            rc_topright = np.array(bottomLeft)
        elif marker_id == 3:
            rc_bottomleft = np.array(topRight)
        else:
            rc_bottomright = np.array(topLeft)
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
            cv2.putText(image, str(
                marker_id), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 2)
            print("[INFO] ArUco marker ID: {}".format(marker_id))

    # TODO: check the homography (whether the camera is facing the target directly)
    roi_width = np.linalg.norm(rc_topright-rc_topleft)/4
    slanted1_center = (rc_topright + rc_bottomright)/2
    slanted1_tr = slanted1_center + [roi_width, -roi_width]
    slanted1_br = slanted1_center + [roi_width, roi_width]
    slanted1_tl = slanted1_center + [-roi_width, -roi_width]
    slanted1_bl = slanted1_center + [-roi_width, roi_width]
    if verbose:
        cv2.imshow("image", image)
        cv2.imshow("extracted edge", image[slanted1_tl[1].astype(int):slanted1_bl[1].astype(
            int), slanted1_tl[0].astype(int): slanted1_tr[0].astype(int)])
        cv2.waitKey(0)

    return image[slanted1_tl[1].astype(int):slanted1_bl[1].astype(int), slanted1_tl[0].astype(int): slanted1_tr[0].astype(int)]


if __name__ == "__main__":
    image = cv2.imread("imgs/reso_01.png")
    #image = cv2.imread("imgs/reso_rotate2.png")

    detect_reso_chart(image, verbose=True)
