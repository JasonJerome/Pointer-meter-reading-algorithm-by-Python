#定义相关的引用
import colorsys
import random
import os
import numpy as np
from yolo import YOLO
from PIL import Image
import cv2

#定义相关的图片及视频的路径
video_path = "D:/test.mp4"
output_path = "D:/0.mp4"
ImageDir = os.listdir("D:/test/testimages")
#用于矩形框的绘制
RecDraw = []


# 用来存储矩形框
# 此段代码为开源项目白嫖代码，用于对每一类产生相应的颜色与之对应
# 很明显开源项目的这段代码也是从yolov3代码白嫖来的
def colors_classes(num_classes):
    if (hasattr(colors_classes, "colors") and
            len(colors_classes.colors) == num_classes):
        return colors_classes.colors

    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    colors_classes.colors = colors  # Save colors for future calls.
    return colors

#这段代码主要是为了完成两个目标框之间的交并比iou的计算
def compute_iou(box1, box2):
    ymin1, xmin1, ymax1, xmax1 = box1
    ymin2, xmin2, ymax2, xmax2 = box2
    # 获取矩形框交集对应的左上角和右下角的坐标（intersection）
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])
    # 计算两个矩形框面积
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
    # 计算交集面积
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)
    # 计算交并比
    return iou


def doiou(boxFilterPerson, boxFilterHat, numPeople, numHat):
    perhat = np.zeros(shape=(numPeople, numHat))
    for perindex in range(numPeople):
        for hatindex in range(numHat):
            perhat[perindex][hatindex] = compute_iou(boxFilterPerson[perindex], boxFilterHat[hatindex])
    return perhat

# track_obj被用作类结构体变量，表征的是被监测物体本身的属性（定义的可以更加简单化一些）
class track_obj(object):
    #def __init__(self,newname)：self.name=newname,通过访问self.name的形式给实例中增加了name变量，并给name赋了初值newname
    def __init__(self):
        self.last_rstate = 0
        self.last_cov = 0
        self.frames = 0
        self.trace_id = 0
        self.color = 0
#该段代码主要是用于计算两个目标框之间的距离
def calDistance(centersPerson, centersHat):
    xAxisDis = np.zeros(shape=(len(centersPerson), len(centersHat)))
    for perindex in range(len(centersPerson)):
        for hatindex in range(len(centersHat)):
            xAxisDis[perindex][hatindex] = centersPerson[perindex][0][0] - centersHat[hatindex][0][0]
    # print(np.fabs(xAxisDis).min(1))
    return(xAxisDis)

# 计算被检测到物体的中点，可以理解为是传感器，能够检测到实际的值
def cal_centre(out_boxes, out_classes, out_scores, score_thres):
    print(out_boxes, out_classes, out_scores)
    dict = {}
    for key in out_classes:
        dict[key] = dict.get(key, 0) + 1
    print(len(dict))
    if len(dict) == 0:
        return 000, 000
    if len(dict) == 1:
        for PersonIndex in range(len(out_boxes)):
            RecDraw.append(out_boxes[PersonIndex])
        return 666, 666
    # elif dict[0] > dict[1]:
    #     return 666, 666     # 简单版本的就是人比帽子多直接警告，复杂一点就直接pass交给后面的任务
    boxFilterPerson = []
    boxFilterHat = []
    centersPerson = []
    centersHat = []
    for div_box, div_class, div_score in zip(out_boxes, out_classes, out_scores):
        if (div_score >= score_thres) and (div_class == 0):
            boxFilterPerson.append(div_box)
            centre = np.array([[(div_box[1] + div_box[3]) // 2], [(div_box[0] + div_box[2]) // 2]])
            centersPerson.append(centre)
        if (div_score >= score_thres) and (div_class == 1):
            boxFilterHat.append(div_box)
            centre_hat = np.array([[(div_box[1] + div_box[3]) // 2], [(div_box[0] + div_box[2]) // 2]])
            centersHat.append(centre_hat)
    numPeople = len(centersPerson)
    numHat = len(centersHat)
    perhat = doiou(boxFilterPerson, boxFilterHat, numPeople, numHat)
    m = perhat.sum(axis=1)
    for index in range(len(m)):
        if m[index] == 0:
            RecDraw.append(boxFilterPerson[index])
            return 666, 666
    xAxisDis = calDistance(centersPerson, centersHat)
    dis = np.fabs(xAxisDis).min(1)
    for dis_index in range(len(dis)):
        if m[dis_index] > 20:
            RecDraw.append(boxFilterPerson[index])
            return 666, 666
    return centersPerson, numPeople


class dokalman():
    def __init__(self):
        self.count = 0
        self.dotracking = []
        self.str_location = []
        self.distance_max = 200

    # """注意矩阵的行列此处并没有进行修改"""
    def tracking(self, centers, num, image, mode):  # 先尝试着单目标的追踪
        font = cv2.FONT_HERSHEY_SIMPLEX
        if mode == 0:
            if centers == 000 and num == 000:
                cv2.putText(image, "need check", (11, 33), font, 1, [230, 0, 0], 2)
            if centers == 666 and num == 666:
                cv2.putText(image, "detecting", (11, 11 + 22), font, 1, [230, 0, 0], 2)
                for index in range(len(RecDraw)):
                    cv2.rectangle(image, (RecDraw[index][1], RecDraw[index][0]), (RecDraw[index][3], RecDraw[index][2]), (230, 0, 0), 2)
            elif centers != 666 or num != 666:
                cv2.putText(image, "need check", (11, 33), font, 1, [230, 0, 0], 2)
        elif mode == 1:
            if centers == 000 and num == 000:
                cv2.putText(image, "need check", (11, 33), font, 1, [0, 0, 230], 2)
            if centers == 666 and num == 666:
                cv2.putText(image, "detecting", (11, 11 + 22), font, 1, [0, 0, 230], 2)
                for index in range(len(RecDraw)):
                    cv2.rectangle(image, (RecDraw[index][1], RecDraw[index][0]), (RecDraw[index][3], RecDraw[index][2]),
                                  (0, 0, 230), 2)
            elif centers != 666 or num != 666:
                cv2.putText(image, "need check", (11, 33), font, 1, [0, 0, 230], 2)


yolov3_args = {
    "model_path": 'logs/000/trained_weights_final.h5',
    "anchors_path": 'model_data/yolo_anchors.txt',
    "classes_path": 'model_data/coco_classes.txt',
    "score": 0.50,
    "iou": 0.3,
    "model_image_size": (416, 416),
    "gpu_num": 1,
}


def image(pic_path):
    mode = 0
    RecDraw.clear()
    if pic_path == 0:
        yolov3 = YOLO(**yolov3_args)
        for i in range(len(ImageDir)):
            RecDraw.clear()
            ImagePath = "D:/test/testimages/" + ImageDir[i]
            ImageName = "D:/test/image/" + str(i) + ".jpg"
            img = Image.open(ImagePath)
            image, boxes, scores, classes = yolov3.detect_image_mul(img)
            centers, num = cal_centre(boxes, classes, scores, 0.05)
            result = np.asarray(image)
            tracker = dokalman()
            tracker.tracking(centers, num, result, mode)
            image_bgr = cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR)
            cv2.imwrite(ImageName, image_bgr)
    elif pic_path != 0:
        yolov3 = YOLO(**yolov3_args)
        img = Image.open(pic_path)
        image, boxes, scores, classes = yolov3.detect_image_mul(img)
        centers, num = cal_centre(boxes, classes, scores, 0.05)
        result = np.asarray(image)
        tracker = dokalman()
        tracker.tracking(centers, num, result, mode)
        # print("look there!", RecDraw)
        image_bgr = cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR)
        cv2.imwrite("D:/test/pp30.jpg", image_bgr)
        cv2.imshow("re", image_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def video():
    mode = 1
    yolov3 = YOLO(**yolov3_args)
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        raise IOError
    video_FourCC = int(video_cap.get(cv2.CAP_PROP_FOURCC))
    video_fps = video_cap.get(cv2.CAP_PROP_FPS)
    video_size = (int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # video_size = (int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    isOutput = True if output_path != "" else False
    if isOutput:
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    tracker = dokalman()

    frame_index = 0
    name = 11632
    while True:
        RecDraw.clear()
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
        # transposedImage = cv2.transpose(frame)
        # flipedImageX = cv2.flip(transposedImage, 0)
        # image_new = Image.fromarray(flipedImageX)
        image_new = Image.fromarray(frame)
        image, boxes, scores, classes = yolov3.detect_image_mul(image_new)
        centers, num = cal_centre(boxes, classes, scores, 0.05)
        result = np.asarray(image)
        tracker.tracking(centers, num, result, mode)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    # print("please input the type of your want to identify")
    # m = input("pic or video? Answer: ")
    # if m == "video":
     video()
    # elif m == "pic":
    #     pic_path = input("please input image path : ")
    #     image(pic_path)
    # image("D:/git/work/keras-yolo3/images/1.jpg")
    # image("D:/r.jpg")
    # image(0)
