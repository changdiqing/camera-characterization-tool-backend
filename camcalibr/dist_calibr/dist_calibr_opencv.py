import numpy as np
import urllib.request
import cv2 as cv


def mse(predictions, targets):
    return ((predictions - targets) ** 2).mean()


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def dist_calibr(images, verbose=False):
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # for fname in img_urls:
    for img in images:
        # for fname in images:
        # img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        found, corners = cv.findChessboardCorners(gray, (7, 7), None)
        # If found, add object points, image points (after refining them)
        if found:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners if verbose
            if verbose:
                img_clone = img.copy()
                cv.drawChessboardCorners(img_clone, (7, 7), corners2, found)
                cv.imshow("img", img_clone)
                cv.waitKey(0)

    # Estimate the calibration parameters
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print(f"imagepoints: {imgpoints}")
    print(f"ret: {ret}")
    print(f"mtx: {mtx}")
    print(f"dist: {type(dist)}")
    print(f"rvecs: {type(rvecs)}")
    print(f"tvecs: {tvecs}")

    # Undistort all images
    calibrated_images = []
    for img in images:

        h, w = img.shape[:2]  # pixel counts
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # undistort
        dst = cv.undistort(img, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y : y + h, x : x + w]
        calibrated_images.append(dst)
        if verbose:
            cv.imshow("calibrated", dst)
            cv.waitKey(0)

    # Compute re-projection error (estimating the accuracy of the calibration parameters)
    error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        # error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        # MSE of the projected points of the ith image
        error = mse(imgpoints[i], imgpoints2)
        error += error
    # RMSE of the projected points of all images
    rms_error = np.sqrt(error / len(objpoints))
    print(f"total error: {rms_error}")

    return mtx.tolist(), dist.tolist(), rvecs, tvecs, rms_error, calibrated_images


if __name__ == "__main__":
    mocked_imgs_small = [
        "imgs/validate_dist_1.png",
        "imgs/validate_dist_2.png",
        "imgs/validate_dist_3.png",
        "imgs/validate_dist_4.png",
        "imgs/validate_dist_5.png",
        "imgs/validate_dist_6.png",
        "imgs/validate_dist_7.png",
        "imgs/validate_dist_8.png",
        "imgs/validate_dist_9.png",
    ]
    mocked_imgs_big = [
        "imgs/validate_dist_big_1.png",
        "imgs/validate_dist_big_2.png",
        "imgs/validate_dist_big_3.png",
        "imgs/validate_dist_big_4.png",
        "imgs/validate_dist_big_5.png",
        "imgs/validate_dist_big_6.png",
        "imgs/validate_dist_big_7.png",
        "imgs/validate_dist_big_8.png",
        "imgs/validate_dist_big_9.png",
    ]
    mocked_imgs3 = ["imgs/validate_dist_big_3.png"]
    mocked_imgs4 = ["imgs/fisheye.jpeg"]

    mocked_urls = [
        "http://127.0.0.1:5000/images/2ab56008-88b3-41df-9b14-0176cc49e020.jpg",
        "http://127.0.0.1:5000/images/16a1f729-2be3-4b13-9a3b-843c971dfa14.jpg",
    ]

    images = []

    # Test with local images
    for fname in mocked_imgs4:
        img = cv.imread(fname)
        images.append(img)

    mtx, dist, rvecs, tvecs, mean_error, calibr_imgs = dist_calibr(images, True)

    print(f"mtx: {mtx}")
    print(f"type of mtx: {type(mtx)}")
    print(f"dist: {dist}")
    print(f"type of dist: {type(dist)}")
    print(f"mean_error: {mean_error}")
    print(f"type of mean_error: {type(mean_error)}")
    # print(f"calibrated results: {calibr_imgs}")
