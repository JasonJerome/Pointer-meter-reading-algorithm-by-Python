import cv2
import numpy as np
from PIL import Image

def degree2num(corrected_img_path):
    """get the class1 pointer degree and map to the number

    :param corrected_img_path: the corrected image path; eg: "./img_test_corrected/test1.png"
    :return: Instrument number
    """
    # read the image and convert to gray image
    gray = cv2.imread(corrected_img_path, 0)

    # Image edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # downsample the image for saving calculating time
    edges_img = Image.fromarray(edges)
    w, h = edges_img.size
    edges_img_resized = edges_img.resize((w // 3, h // 3))
    edges_img_resized_array = np.array(edges_img_resized)

    # use Hough Circle Transform to detect the dashboard of reduced images
    circles = cv2.HoughCircles(edges_img_resized_array, cv2.HOUGH_GRADIENT, 1, 100,
                               param1=150, param2=100, minRadius=0, maxRadius=0)
    circles_int = np.uint16(np.around(circles)) # for visualizing
    x, y, _ = circles[0][0]  # suppose to find the biggest cycle ！！！！！！！！
    x, y = x * 3, y * 3  # map the cycle center to source image

    # detect the lines
    minLineLength = 120
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap).squeeze(1)

    """Detect the pointer line using a prior conditions: 
        1. a straight line passes through the cycle center; 
        2. the length of the line segment of the pointer is the longest
    """
    current_lines = []
    for x1, y1, x2, y2 in lines:
        # pass through the cycle center
        error = np.abs((y2 - y) * (x1 - x) - (y1 - y) * (x2 - x))
        if error < 1000:  # can change the threshold ！！！！！！
            current_lines.append((x1, y1, x2, y2))
            # for visualizing
            # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # find the longest line
    pointer_line = ()
    pointer_length = 0
    for x1, y1, x2, y2 in current_lines:
        length = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
        if length > pointer_length:
            pointer_length = length
            pointer_line = (x1, y1, x2, y2)

    # for visualizing
    x1, y1, x2, y2 = pointer_line
    cv2.line(gray, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # compute the pointer degree
    pointer_grad = np.abs(x2 - x1) / np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    poiner_degree = np.arccos(pointer_grad) / np.pi * 180

    # The center of the circle is compared to determine
    # the position of the pointer and then obtain the real pointer degree
    if x1 > x and y1 < y:  # In the first quadrant
        poiner_degree = poiner_degree
    elif x1 < x and y1 < y:  # In the second quadrant
        poiner_degree = 180 - poiner_degree
    elif x1 < x and y1 > y:  # In the third quadrant
        poiner_degree = 180 + poiner_degree
    else:                    # In the fourth quadrant
        poiner_degree = 360 - poiner_degree

    # map the degree to num
    num = 0.56 # from the map (poiner_degree to num)

    # for visualizing
    for i in circles_int[0, :]:
        # draw the outer circle
        cv2.circle(edges_img_resized_array, (i[0], i[1]), i[2], (255, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(edges_img_resized_array, (i[0], i[1]), 2, (255, 0, 0), 3)

    # show the result
    cv2.imshow("edges", edges)
    cv2.imshow("img", gray)
    cv2.imshow("edges_resized", edges_img_resized_array)
    cv2.waitKey(0)

    return num

if __name__ == "__main__":
    corrected_img_path = "../img_test_corrected/test1.png"
    degree = degree2num(corrected_img_path)
    print(degree)