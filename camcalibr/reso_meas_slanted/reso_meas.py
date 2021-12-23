import sys
import cv2
import numpy as np
from .detect_reso_chart import detect_reso_chart
from .compute_mtf import compute_mtf


def reso_meas(images, verbose=False):
    """ measure the resolution from multiple photos of resolution test chart
        STEP1: Extract edge images from photos.
        STEP2: Estimate MTF from edge images
        STEP3: Merge the detected  MTFs and return

    Args:
        images ([numpy.array]): Photo of resolution test chart
        verbose (bool, optional): Defaults to False.

    Return:
        (touple): x and y coordinates of MTF curve
        ([numpy.array]): output images
    """

    mtf_list = []
    for img in images:
        edge = detect_reso_chart(img, verbose)
        mtf = compute_mtf(edge, verbose)
        mtf_list.append(mtf)
        """
        try:
            edge = detect_reso_chart(img, verbose)
            mtf = compute_mtf(edge, verbose=False)
            print(f"mtf: {mtf}")
            mtf_list.append(mtf)
        except:
            continue
        """

    if len(mtf_list) == 0:
        sys.exit("[ERROR]: no valid image for resolution measurement!")

    mtf_x = mtf_list[0][0]
    mtf_y = mtf_list[0][1] if len(
        mtf_list) == 1 else np.mean([mtf[1] for mtf in mtf_list], axis=0)

    return (mtf_x, mtf_y), []


if __name__ == "__main__":
    mocked_urls = [
        'test_chart/reso_test_chart.png'
    ]

    mocked_imgs = []
    for url in mocked_urls:
        mocked_imgs.append(cv2.imread(url, 0))

    mtf, output_imgs = reso_meas(mocked_imgs, verbose=True)
    print(mtf)
