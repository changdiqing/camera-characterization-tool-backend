from color_calibr_ccm.colorchecker_detection import detect_colorchecker
import cv2

# filename = "color_calibr_ccm/imgs/color_ref_s.jpg"
# filename = "color_calibr_ccm/imgs/src_chair.png"
filename = "color_calibr_ccm/imgs/ref_chair.png"
img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
cv2.imshow("test", img)
cv2.waitKey(0)
detect_colorchecker(img, 4, 6, "DICT_5X5_50", verbose=True)
