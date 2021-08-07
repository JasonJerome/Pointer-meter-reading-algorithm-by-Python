import cv2
import numpy as np
import imutils
from PIL import Image
def degree2num(corrected_img_path):
    """Get the number of second pointer meter

    :param corrected_img_path: the path of test image
    :return: the num or None
    """
    """ to detect the rectangle """
    # load the image and resize it to a smaller factor so that
    # the shapes can be approximated better
    img = cv2.imread(corrected_img_path, 0)
    resized_gray = imutils.resize(img, width=300)
    ratio = img.shape[0] / float(resized_gray.shape[0])

    # blur it slightly, and threshold it
    blurred = cv2.bilateralFilter(resized_gray, 11, 17, 17)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    # find contours in the thresholded image
    cants = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cants = imutils.grab_contours(cants)

    # detect the rectangle: 1. have four points 2. the area is biggest
    rectangle = None
    biggest_area = 0
    for cant in cants:
        peri = cv2.arcLength(cant, True)
        approx = cv2.approxPolyDP(cant, 0.03 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(cant)
            if area > biggest_area:
                rectangle = {"cant": [cant], "approx": approx}
                biggest_area = area

    """ detect the pointer and computer the degree, then using the map(degree to num) to find the num"""

    # crop the rectangle
    points = rectangle["approx"]
    y1, y2, x1, x2 = int(points[0][0][0] * ratio), int(points[2][0][0] * ratio), int(points[0][0][1] * ratio), int(points[2][0][1] * ratio)
    img_rectangele_cut = img[x1:x2, y1:y2]
    img_rectangele_cut_blurred = cv2.bilateralFilter(img_rectangele_cut, 11, 17, 17)

    # Image edge detection
    edges = cv2.Canny(img_rectangele_cut_blurred, 100, 150, apertureSize=3)
    # detect the lines
    minLineLength = 300
    maxLineGap = 25
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength, maxLineGap).squeeze(1)

    # to show
    current_lines = []
    for x1, y1, x2, y2 in lines:
        # remove the surrounding lines
        if y2 - y1 > 10:
            if x1 > 10 and x1 < img_rectangele_cut.shape[0] - 10:  # !!!!! can change
                current_lines.append((x1, y1, x2, y2))
                # for show
                cv2.line(img_rectangele_cut_blurred, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # compute the pointer degree
    x1, y1, x2, y2 = current_lines[0]
    pointer_grad = np.abs(x2 - x1) / np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    poiner_degree = np.arccos(pointer_grad) / np.pi * 180

    if y2 > y1:
        poiner_degree = 180 - poiner_degree

    # map the degree to num
    num = 0.33  # from the map (poiner_degree to num)

    # to show
    for key in range(len(rectangle["cant"])):
        cv2.drawContours(resized_gray, rectangle["cant"], key, (0, 255, 0), 3)

    #cv2.imshow("gray0", resized_gray)
    #cv2.imshow("gray1", img)
    #cv2.imshow("gray2", edges)
    cv2.imshow("gray3", img_rectangele_cut_blurred)
    cv2.waitKey(0)
    return num

if __name__ == "__main__":
    degree2num("../template/class2.png")