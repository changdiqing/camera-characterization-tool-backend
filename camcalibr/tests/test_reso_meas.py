from camcalibr.reso_meas_slanted.reso_meas import reso_meas
import cv2

mocked_urls = ["imgs/reso_test_chart.png"]

mocked_imgs = []
for url in mocked_urls:
    mocked_imgs.append(cv2.imread(url, 0))

mtf, output_imgs = reso_meas(mocked_imgs, verbose=True)
print(mtf)
