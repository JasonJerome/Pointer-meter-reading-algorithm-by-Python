"""
    created by:maogu123@126.com
    data:2021-01-04

    功能：在视频中寻找匹配的仪表，并识别指针的角度
"""
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt


class C_ammerter:
    def __init__(self,temp):
        # 获取模板样本
        self.template = temp
        # 基于视频流
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # 获取模板的尺寸
        self.w =  self.template.shape[0]
        self.h =  self.template.shape[1]
        
        methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF',
                   'cv2.TM_SQDIFF_NORMED']
        # 平方差 SQDIFF
        # 相关匹配 CCORR
        # 相关系数法 CCOEFF
        self.method = cv2.TM_CCORR


    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()






    # 获取匹配的图片位置
    def get_match(self,img):
        res = cv2.matchTemplate(img, self.template, self.method)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        print("----------------------------" )
        print("min_val", min_val)
        print("max_val", max_val)
        print("min_loc", min_loc)
        print("max_loc", max_loc)
        print("----------------------------" )
        bottom_right = (top_left[0] +  self.w, top_left[1] +  self.h)
        cv2.rectangle(img, top_left, bottom_right, 255, 2)
        c_x, c_y = ((np.array(top_left) + np.array(bottom_right)) / 2).astype(np.int)
        # print(c_x, c_y)
        return max_val,top_left, bottom_right

        #   return img[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w]


    def am_run(self):
        while True:
            ret, frame = self.cap.read()
            if frame is None:
                print('video picture is none  --continue ')
                continue

            gray = frame.copy()
            # cv2.imshow('origin', gray)

            # 匹配模板 框出匹配区域
            image = gray.copy()
            maxval,t_left, b_right = self.get_match(gray)
            if maxval < 16000000000:  # 对匹配程度做判断
                print("---------------------------------------")
                print('matchTemplate is not enough  --continue')
                print("---------------------------------------")
                result =frame
                image=frame
            else:

                cv2.rectangle(image, t_left, b_right, 255, 2)



                # 高斯除噪
                kernel = np.ones((6,6), np.float32) / 36
                gray_cut_filter2D = cv2.filter2D(image[t_left[1]:t_left[1] +  self.h, t_left[0]:t_left[0] +  self.w], -1, kernel)

                # 灰度图 二值化
                gray_img = cv2.cvtColor(gray_cut_filter2D, cv2.COLOR_BGR2GRAY)
                ret, thresh1 = cv2.threshold(gray_img, 130, 255, cv2.THRESH_BINARY)

                # 二值化后 分割主要区域 减小干扰 模板图尺寸371*369
                tm = thresh1.copy()
                test_main = tm[50:500, 50:500]

                # 边缘化检测
                edges = cv2.Canny(test_main, 50, 150, apertureSize=3)

                # 霍夫直线
                lines = cv2.HoughLines(edges, 1, np.pi / 180, 60)
                if lines is None:
                    continue
                result = edges.copy()

                for line in lines[0]:
                    rho = line[0]  # 第一个元素是距离rho
                    theta = line[1]  # 第二个元素是角度theta
                    print('distance:' + str(rho), 'theta:' + str(((theta / np.pi) * 180)))
                    lbael_text = 'distance:' + str(round(rho))+  'theta:' + str(round((theta / np.pi) * 180-90,2))
                    cv2.putText(image, lbael_text,(t_left[0],t_left[1]-12),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                    if (theta > 3 * (np.pi / 3)) or (theta < (np.pi / 2)):  # 垂直直线
                        # 该直线与第一行的交点
                        pt1 = (int(rho / np.cos(theta)), 0)
                        # 该直线与最后一行的焦点
                        pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
                        # 绘制一条白线
                        cv2.line(result, pt1, pt2,255, 1)
                        # print('theat >180 theta<90')

                    else:  # 水平直线
                        # 该直线与第一列的交点
                        pt1 = (0, int(rho / np.sin(theta)))
                        # 该直线与最后一列的交点
                        pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
                        # 绘制一条直线
                        cv2.line(result, pt1, pt2, 255, 1)
                        # print('theat <180 theta > 90')

            # 直线拟合
            cv2.imshow('result', result)
            cv2.imshow('rectangle', image)
            if cv2.waitKey(1) & 0XFF == ord('q'):
                break


        # Terminal进入当前文件  命令生成exe: pyinstaller --console --onefile ammeter.py
