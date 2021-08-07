"""
    created by:maogu123@126.com

    data:2021-01-06

    功能：在视频中,抓取目标仪表的轮廓，按轮廓切割 ，然后霍夫直线得到指针的角度

"""
from typing import Optional, Any

import cv2
import numpy as np
import time
from matplotlib import pyplot as plt


# 获取面积最大的轮廓数据
def max_contours(contour):
    area = map(cv2.contourArea, contour)
    area_list = list(area)
    get_area_max = max(area_list)
    get_post = area_list.index(get_area_max)
    return get_post, get_area_max


# 将直线延长与边界相交， 在图形中画出
def get_HImg(H_image, Lines):
    for Line in Lines[0]:
        Rho = Line[0]  # 第一个元素是距离rho
        Theta = Line[1]  # 第二个元素是角度theta

        print('theta:' + str(((Theta / np.pi) * 180)))
        if (Theta > 3 * (np.pi / 3)) or (Theta < (np.pi / 2)):  # 垂直直线
            # 该直线与第一行的交点
            Pt1 = (int(Rho / np.cos(Theta)), 0)
            # 该直线与最后一行的焦点
            Pt2 = (int((Rho - H_image.shape[0] * np.sin(Theta)) / np.cos(Theta)), H_image.shape[0])
            # 绘制一条线
            cv2.line(H_image, Pt1, Pt2, 255, 1)

        else:  # 水平直线
            # 该直线与第一列的交点
            Pt1 = (0, int(Rho / np.sin(Theta)))
            # 该直线与最后一列的交点
            Pt2 = (H_image.shape[1], int((Rho - H_image.shape[1] * np.cos(Theta)) / np.sin(Theta)))
            # 绘制一条直线
            cv2.line(H_image, Pt1, Pt2, 255, 1)

        return H_image



cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('frame is none  --continue')
        continue
    # 保存原图片
    origin = frame

    # 高斯除噪
    kernel = np.ones((5, 5), np.float32) / 25
    gray_cut_filter2D = cv2.filter2D(frame, -1, kernel)

    # 转为灰度图。再二值化
    img_gray = cv2.cvtColor(gray_cut_filter2D, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)  # THRESH_BINARY_INV
    # cv2.imshow('thresh', thresh)

    # 查找轮廓
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)  # cv2.CHAIN_APPROX_NONE CHAIN_APPROX_SIMPLE

    # 查找面积最大的轮廓
    post, area_max = max_contours(contours)
    print('area:' + str(area_max))

    # 过滤面积小的区域
    if area_max < 145000:
        print('area_max is not enough --continue')
        continue
    else:
        # 在原图上画出轮廓
        C_img = cv2.drawContours(frame, contours, post, (0, 255, 0), 1)
        # cv2.imshow('C_img', C_img)

    # 新建空白图像，放入轮廓内的图像
    cimg = np.zeros_like(frame)
    cimg[:, :, :] = 255
    cv2.drawContours(cimg, contours, post, (0, 0, 0), -1)

    # 抓取后的图像
    final = cv2.bitwise_or(frame, cimg)

    # 高斯除噪 灰度图 二值化 边缘化检测
    final_filter2D = cv2.filter2D(final, -1, kernel)
    final_gray = cv2.cvtColor(final_filter2D, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(final_gray, 80, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresh1, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 60)

    if lines is None or len(lines) < 1:
        continue

    Line = lines[0][0]
    Rho = Line[0]  # 第一个元素是距离rho
    Theta = Line[1]  # 第二个元素是角度theta
    print('------------------------')
    print('Line:', Line)
    print('------------------------')

    lbael_text = 'distance:' + str(round(Rho)) + 'theta:' + str(round((Theta / np.pi) * 180 - 90, 2))
    cv2.putText(frame, lbael_text, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    result = get_HImg(frame, lines)  # edges


    cv2.imshow('result', result)

    gray_result = get_HImg(edges, lines)
    cv2.imshow('gray_result', gray_result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Terminal进入当前文件  命令生成exe: pyinstaller --console --onefile ammeter.py
