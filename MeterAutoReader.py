import colorsys
import random
import os
import numpy as np
from yolo import YOLO
from PIL import Image
import cv2
import math
#import cv2 as cv
#import argparse
import matplotlib.pyplot as plt

video_path = "D:/test.mp4"
output_path = "D:/0.mp4"
ImageDir = os.listdir("D:/test/testimages")
j = 0
a = 0
b = 0
c = 0
detected_theata = 0
detected_theata1 = 0
detected_theata2 = 0
detected_theata3 = 0
jiaodu = 0


#这一步是为了调用已经训练好的Yolov3模型参数
yolov3_args = {
    "model_path": 'logs/000/trained_weights_final.h5',
    "anchors_path": 'model_data/yolo_anchors.txt',
    "classes_path": 'model_data/coco_classes.txt',
    "score": 0.08,
    "iou": 0.3,
    "model_image_size": (416, 416),
    "gpu_num": 1,
}

def image(pic_path):
    if pic_path == 0:
        yolov3 = YOLO(**yolov3_args)
        for i in range(len(ImageDir)):
            ImagePath = "D:/test/testimages/" + ImageDir[i]
            ImageName = "D:/test/testimages/" + str(i) + ".jpg"
            img = Image.open(ImagePath)
            image, boxes, scores, classes = yolov3.detect_image_mul(img)
            origin = np.asarray(image) #将数据转为矩阵
            image_bgr = cv2.cvtColor(np.asarray(origin), cv2.COLOR_RGB2BGR)#cv2下的色彩空间灰度化
            cv2.imwrite(ImageName, image_bgr)
    elif pic_path != 0:
        yolov3 = YOLO(**yolov3_args)
        img = Image.open(pic_path)#打开图片
        img2 = cv2.imread(pic_path)
        image, boxes, scores, classes = yolov3.detect_image_mul(img)#yolov3检测
        origin = np.asarray(image)  # 将数据转为矩阵
        image_bgr = cv2.cvtColor(np.asarray(origin), cv2.COLOR_RGB2BGR)  # cv2下的色彩空间灰度化
        cv2.imwrite("D:/git/work/keras-yolo3/kuangxuanimages/detected.jpg", image_bgr)
        #boxes内返回的是yolo预测出来的边框坐标，通过该坐标可以对原图像进行裁剪
        for i in range(boxes.shape[0]):
            top, left, bottom, right = boxes[i]
            # 或者用下面这句等价
            #top = boxes[0][0]
            #left = boxes[0][1]
            #bottom = boxes[0][2]
            #right = boxes[0][3]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5
            # 左上角点的坐标
            top = int(max(0, np.floor(top + 0.5).astype('int32')))
            left = int(max(0, np.floor(left + 0.5).astype('int32')))
            # 右下角点的坐标
            bottom = int(min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32')))
            right = int(min(np.shape(image)[1], np.floor(right + 0.5).astype('int32')))
            # 记录图片的高度与宽度
            a = bottom - top
            b = right - left
            print ('height', a)
            print ('with', b)
            croped_region = image_bgr[top:bottom, left:right]  # 先高后宽
            #cv2.imshow("cropimage", croped_region)
            # 将裁剪好的目标保存到本地
            j + 1
            cv2.imwrite("D:/git/work/keras-yolo3/kuangxuanimages/cutted_img_"+str(j)+".jpg", croped_region)
            print('cropped successed')

            cv2.waitKey(0)
            cv2.destroyAllWindows()


def vameterdetect(num):
   if num == 1:


        origin = cv2.imread("D:/git/work/keras-yolo3/kuangxuanimages/cutted_img_"+str(j)+".jpg", 0)
        nor = cv2.resize(origin, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)#图片归一化cv2.resize（输入图片，输出图片，沿x轴缩放系数，沿y轴缩放系数，插入方式为双线性插值（默认方式））

        image_bgr = cv2.cvtColor(nor, cv2.COLOR_RGB2BGR)#转换为灰度图
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        median = cv2.medianBlur(origin, 1)# 中值滤波去噪cv2.medianBlur(原图片, 当前的方框尺寸)

        edges = cv2.Canny(median, 250, 350, apertureSize=3)# 边缘检测cv2.Canny（原图片， 最小阈值，最大阈值，Sobel算子的大小）
        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形结构
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))   # 椭圆结构
        # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))  #十字结构
        # cv2.getStructuringElement(指定形状，内核的尺寸，锚点的位置 ) 返回指定形状和尺寸的结构元素。

        # 霍夫直线
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 60)
        result = edges.copy()
        for line in lines[5]:
            rho = line[0]  # 第一个元素是距离rho
            theta = line[1]  # 第二个元素是角度theta
            detected_theata1 = ((theta / np.pi) * 180)
            print('distance:' + str(rho), 'theta:' + str(((theta / np.pi) * 180)))
            lbael_text = 'distance:' + str(round(rho)) + 'theta:' + str(round((theta / np.pi) * 180 - 90, 2))
            if (theta > 3 * (np.pi / 3)) or (theta < (np.pi / 2)):  # 垂直直线
                # 该直线与第一行的交点
                pt1 = (int(rho / np.cos(theta)), 0)
                # 该直线与最后一行的焦点
                pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
                # 绘制一条白线
                cv2.line(result, pt1, pt2, (255, 0, 0), 2, cv2.LINE_AA)
                # print('theat >180 theta<90')

            else:  # 水平直线
                # 该直线与第一列的交点
                pt1 = (0, int(rho / np.sin(theta)))
                # 该直线与最后一列的交点
                pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
                # 绘制一条直线
                cv2.line(result, pt1, pt2, (255, 0, 0), 2, cv2.LINE_AA)
                # print('theat <180 theta > 90')

        for line in lines[18]:
            rho = line[0]  # 第一个元素是距离rho
            theta = line[1]  # 第二个元素是角度theta
            detected_theata2 = ((theta / np.pi) * 180)
            print('distance:' + str(rho), 'theta:' + str(((theta / np.pi) * 180)))
            lbael_text = 'distance:' + str(round(rho)) + 'theta:' + str(round((theta / np.pi) * 180 - 90, 2))
            if (theta > 3 * (np.pi / 3)) or (theta < (np.pi / 2)):  # 垂直直线
                # 该直线与第一行的交点
                pt1 = (int(rho / np.cos(theta)), 0)
                # 该直线与最后一行的焦点
                pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
                # 绘制一条白线
                cv2.line(result, pt1, pt2, (255, 0, 0), 2, cv2.LINE_AA)
                # print('theat >180 theta<90')

            else:  # 水平直线
                # 该直线与第一列的交点
                pt1 = (0, int(rho / np.sin(theta)))
                # 该直线与最后一列的交点
                pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
                # 绘制一条直线
                cv2.line(result, pt1, pt2, (255, 0, 0), 2, cv2.LINE_AA)
                # print('theat <180 theta > 90')

        for line in lines[4]:
            rho = line[0]  # 第一个元素是距离rho
            theta = line[1]  # 第二个元素是角度theta
            detected_theata3 = ((theta / np.pi) * 180)
            print('distance:' + str(rho), 'theta:' + str(((theta / np.pi) * 180)))
            lbael_text = 'distance:' + str(round(rho)) + 'theta:' + str(round((theta / np.pi) * 180 - 90, 2))
            if (theta > 3 * (np.pi / 3)) or (theta < (np.pi / 2)):  # 垂直直线
                # 该直线与第一行的交点
                pt1 = (int(rho / np.cos(theta)), 0)
                # 该直线与最后一行的焦点
                pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
                # 绘制一条白线
                cv2.line(result, pt1, pt2, (255, 0, 0), 2, cv2.LINE_AA)
                # print('theat >180 theta<90')

            else:  # 水平直线
                # 该直线与第一列的交点
                pt1 = (0, int(rho / np.sin(theta)))
                # 该直线与最后一列的交点
                pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
                # 绘制一条直线
                cv2.line(result, pt1, pt2, (255, 0, 0), 2, cv2.LINE_AA)
                # print('theat <180 theta > 90')



        #cv2.imwrite("D:/git/work/keras-yolo3/kuangxuanimages/median.jpg", median)
        cv2.imwrite("D:/git/work/keras-yolo3/kuangxuanimages/edge.jpg", edges)
        cv2.imwrite("D:/git/work/keras-yolo3/kuangxuanimages/result.jpg", result)



        #detected_theata = ((detected_theata2 - detected_theata3) / (detected_theata3 - detected_theata1)) * 800
        #detected_theata = ((detected_theata1 - detected_theata3) / (detected_theata2 - detected_theata3)) * 500
        #detected_theata = ((detected_theata2 - detected_theata1 + 180) / (detected_theata3 - detected_theata1 + 180)) * 2.5
        #detected_theata = ((detected_theata1 - detected_theata2) / (detected_theata3 - detected_theata2 + 180)) * 120 - 10
        #detected_theata = (180 - (detected_theata2 - detected_theata1)) / (360 - (detected_theata2 - detected_theata3)) * 1
        detected_theata = ((180 + detected_theata3 - detected_theata1)) / (360 - (detected_theata1 - detected_theata2)) * 1.6 + 0.03

   return detected_theata

def caculatejiaodu(num):
    if num == 1 :
        jiaodu = vameterdetect(1)
        print('readnum = ', jiaodu)
        image_detected = cv2.imread("D:/git/work/keras-yolo3/kuangxuanimages/detected.jpg", 0)
        #image_cov = cv2.cvtColor(image_detected, cv2.COLOR_GRAY2BGR)
        cv2.putText(image_detected, 'Readnum = {}'.format(jiaodu), (11, 11 + 22), cv2.FONT_HERSHEY_COMPLEX, 1, [230, 0, 0], 2)
        cv2.imwrite("D:/git/work/keras-yolo3/kuangxuanimages/read_num.jpg", image_detected)
        cv2.imshow("ReadNum", image_detected)
        print('Read success!')
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return jiaodu


def video():
    #jiaodu = caculatejiaodu(1)
    #mode = 1
    yolov3 = YOLO(**yolov3_args)
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        raise IOError
    video_FourCC = int(video_cap.get(cv2.CAP_PROP_FOURCC))
    video_fps = video_cap.get(cv2.CAP_PROP_FPS)
    video_size = (int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)

    frame_index = 0
    name = 4228
    while True:
        #RecDraw.clear()
        return_value, frame = video_cap.read()
        frame_index = frame_index + 1
        if frame is None:
            break
        if frame_index % 2 == 1:
            x, y = frame.shape[0:2]
            new_image = cv2.resize(frame, (int(y / 2), int(x / 2)))
            name += 1
            strname = "D:/test/" + str(name) + ".jpg"
            cv2.imwrite(strname, new_image)
        image_new = Image.fromarray(frame)
        image, boxes, scores, classes = yolov3.detect_image_mul(image_new)
        origin = np.asarray(image)
        image_bgr = cv2.cvtColor(np.asarray(origin), cv2.COLOR_RGB2BGR)  # cv2下的色彩空间灰度化
        cv2.imwrite("D:/git/work/keras-yolo3/kuangxuanimages/detected.jpg", image_bgr)
        # boxes内返回的是yolo预测出来的边框坐标，通过该坐标可以对原图像进行裁剪
        for i in range(boxes.shape[0]):
            top, left, bottom, right = boxes[i]
            # 或者用下面这句等价
            # top = boxes[0][0]
            # left = boxes[0][1]
            # bottom = boxes[0][2]
            # right = boxes[0][3]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5
            # 左上角点的坐标
            top = int(max(0, np.floor(top + 0.5).astype('int32')))
            left = int(max(0, np.floor(left + 0.5).astype('int32')))
            # 右下角点的坐标
            bottom = int(min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32')))
            right = int(min(np.shape(image)[1], np.floor(right + 0.5).astype('int32')))
            # 记录图片的高度与宽度
            a = bottom - top
            b = right - left
            print('height', a)
            print('with', b)
            croped_region = image_bgr[top:bottom, left:right]  # 先高后宽
            # cv2.imshow("cropimage", croped_region)

            nor = cv2.resize(croped_region, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)  # 图片归一化cv2.resize（输入图片，输出图片，沿x轴缩放系数，沿y轴缩放系数，插入方式为双线性插值（默认方式））

            image_bgr = cv2.cvtColor(nor, cv2.COLOR_RGB2BGR)  # 转换为灰度图
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

            median = cv2.medianBlur(origin, 1)  # 中值滤波去噪cv2.medianBlur(原图片, 当前的方框尺寸)

            edges = cv2.Canny(median, 250, 350, apertureSize=3)  # 边缘检测cv2.Canny（原图片， 最小阈值，最大阈值，Sobel算子的大小）
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形结构
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 椭圆结构
            # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))  #十字结构
            # cv2.getStructuringElement(指定形状，内核的尺寸，锚点的位置 ) 返回指定形状和尺寸的结构元素。

            # 霍夫直线
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 60)
            result = edges.copy()
            for line in lines[5]:
                rho = line[0]  # 第一个元素是距离rho
                theta = line[1]  # 第二个元素是角度theta
                detected_theata1 = ((theta / np.pi) * 180)
                print('distance:' + str(rho), 'theta:' + str(((theta / np.pi) * 180)))
                lbael_text = 'distance:' + str(round(rho)) + 'theta:' + str(round((theta / np.pi) * 180 - 90, 2))
                if (theta > 3 * (np.pi / 3)) or (theta < (np.pi / 2)):  # 垂直直线
                    # 该直线与第一行的交点
                    pt1 = (int(rho / np.cos(theta)), 0)
                    # 该直线与最后一行的焦点
                    pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
                    # 绘制一条白线
                    cv2.line(result, pt1, pt2, (255, 0, 0), 2, cv2.LINE_AA)
                    # print('theat >180 theta<90')

                else:  # 水平直线
                    # 该直线与第一列的交点
                    pt1 = (0, int(rho / np.sin(theta)))
                    # 该直线与最后一列的交点
                    pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
                    # 绘制一条直线
                    cv2.line(result, pt1, pt2, (255, 0, 0), 2, cv2.LINE_AA)
                    # print('theat <180 theta > 90')

            for line in lines[18]:
                rho = line[0]  # 第一个元素是距离rho
                theta = line[1]  # 第二个元素是角度theta
                detected_theata2 = ((theta / np.pi) * 180)
                print('distance:' + str(rho), 'theta:' + str(((theta / np.pi) * 180)))
                lbael_text = 'distance:' + str(round(rho)) + 'theta:' + str(round((theta / np.pi) * 180 - 90, 2))
                if (theta > 3 * (np.pi / 3)) or (theta < (np.pi / 2)):  # 垂直直线
                    # 该直线与第一行的交点
                    pt1 = (int(rho / np.cos(theta)), 0)
                    # 该直线与最后一行的焦点
                    pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
                    # 绘制一条白线
                    cv2.line(result, pt1, pt2, (255, 0, 0), 2, cv2.LINE_AA)
                    # print('theat >180 theta<90')

                else:  # 水平直线
                    # 该直线与第一列的交点
                    pt1 = (0, int(rho / np.sin(theta)))
                    # 该直线与最后一列的交点
                    pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
                    # 绘制一条直线
                    cv2.line(result, pt1, pt2, (255, 0, 0), 2, cv2.LINE_AA)
                    # print('theat <180 theta > 90')

            for line in lines[4]:
                rho = line[0]  # 第一个元素是距离rho
                theta = line[1]  # 第二个元素是角度theta
                detected_theata3 = ((theta / np.pi) * 180)
                print('distance:' + str(rho), 'theta:' + str(((theta / np.pi) * 180)))
                lbael_text = 'distance:' + str(round(rho)) + 'theta:' + str(round((theta / np.pi) * 180 - 90, 2))
                if (theta > 3 * (np.pi / 3)) or (theta < (np.pi / 2)):  # 垂直直线
                    # 该直线与第一行的交点
                    pt1 = (int(rho / np.cos(theta)), 0)
                    # 该直线与最后一行的焦点
                    pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
                    # 绘制一条白线
                    cv2.line(result, pt1, pt2, (255, 0, 0), 2, cv2.LINE_AA)
                    # print('theat >180 theta<90')

                else:  # 水平直线
                    # 该直线与第一列的交点
                    pt1 = (0, int(rho / np.sin(theta)))
                    # 该直线与最后一列的交点
                    pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
                    # 绘制一条直线
                    cv2.line(result, pt1, pt2, (255, 0, 0), 2, cv2.LINE_AA)
                    # print('theat <180 theta > 90')

        detected_theata = ((180 + detected_theata3 - detected_theata1)) / (360 - (detected_theata1 - detected_theata2)) * 1.6 + 0.29

        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.putText(origin, 'Readnum = {}'.format(detected_theata), (11, 11 + 22), cv2.FONT_HERSHEY_COMPLEX, 1, [230, 0, 0], 2)
        cv2.imshow("result", origin)
        if isOutput:
            out.write(origin)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




if __name__ == '__main__':
    # print("please input the type of your want to identify")
    # m = input("pic or video? Answer: ")
    # if m == "video":image
    # elif m == "pic":
    #     pic_path = input("please input image path : ")
    #     image(pic_path)
    #image("D:/git/work/keras-yolo3/images/6959.jpg")
    #meterdetect(1)
    #vameterdetect(1)
    caculatejiaodu(1)
    #video()
    # image("D:/r.jpg")
    # image(0)


