""" A show case of using CCM3x3 to calibrate img/input.png
"""

from cv2 import cv2
from .CCM_3x3 import CCM_3x3
from .colorchecker_detection import detect_colorchecker


def color_calibration(
    src_imgs,
    src_color_space,
    src_is_linear,
    ref_imgs,
    ref_color_space,
    ref_is_linear,
    verbose=False,
    distance="de00",
):
    """Function that does color calibration for a given target image according to a given reference image

    STEP1: load the colorcheckers from the src and ref images
    STEP2: TODO: linearize the src and ref color checkers if necessary
    STEP3: TODO: convert the src and ref color checkers into the same color space (usually the color space of the ref image)
    STEP4: optimize the CCM to minimize the CIE2000 distance between the ref and calibrated target color checkers
    STEP5: compute the calibrated image with the optimzed CCM

    Args:
        src (String): path of target image file
        src_color_space (enum color_space): color space of target image
        src_is_linear (bool): indicates whether the target image is linearized (sRGB or RGB)
        ref (String): path of reference image file
        ref_color_space (enum color_space): color space of reference image
        ref_is_linear (bool): indicates whether the reference iamge is linearized (sRGB or RGB)
    """

    # Paramters of the standard color checker with aruco tags
    col_n = 6
    row_n = 4
    aruco_type = "DICT_5X5_50"

    # load the colorcheckers from the src and ref images
    src_colorchecker = None
    ref_colorchecker = None
    for img in src_imgs:
        try:
            color_checker = detect_colorchecker(img, row_n, col_n, aruco_type, verbose=verbose)
        except SystemExit:
            continue

        if src_colorchecker is None:
            src_colorchecker = color_checker
        src_colorchecker = (src_colorchecker + color_checker) / 2

    for img in ref_imgs:
        try:
            color_checker = detect_colorchecker(img, row_n, col_n, aruco_type, verbose=verbose)
        except SystemExit:
            continue

        if ref_colorchecker is None:
            ref_colorchecker = color_checker
        ref_colorchecker = (ref_colorchecker + color_checker) / 2

    # TODO: if the src has a different color space than the ref image, unify their color spaces

    # use CCM_3x3 to find the optimized CCM, which brings src closer to ref
    ccm = CCM_3x3(src_colorchecker, ref_colorchecker, distance=distance)
    ccm_3x3, error = ccm.value()

    calibrated_images = []
    for img in src_imgs:
        img = ccm.infer_image(img)
        calibrated_images.append(img)
        if verbose:
            cv2.imshow("image after calibration", img)
            cv2.imwrite("imgs/output_infered.png", img)

    if verbose:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return ccm_3x3, error, calibrated_images


if __name__ == "__main__":
    src = [
        # "imgs/input1_aruco_1.png",
        # "imgs/input1_aruco.png"
        "imgs/color_test_chart_gray.png"
    ]
    src_color_space = "sRGB"
    src_is_linear = False

    ref = [
        # "imgs/input1_aruco_ref.png",
        # "imgs/input1_aruco_ref2.png",
        # "imgs/input1_aruco_ref3.png"
        "imgs/color_test_chart_gray.png"
    ]
    ref_color_space = "sRGB"
    ref_is_linear = False

    src_imgs = []
    for img_url in src:
        img = cv2.imread(img_url)
        src_imgs.append(img)

    ref_imgs = []
    for img_url in ref:
        img = cv2.imread(img_url)
        ref_imgs.append(img)

    ccm, error, calibr_imgs = color_calibration(
        src_imgs, src_color_space, src_is_linear, ref_imgs, ref_color_space, ref_is_linear, verbose=True
    )

    print(f"ccm: {ccm}")
    print(f"error: {error}")
