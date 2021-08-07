import cv2
from ammeter import C_ammerter


if __name__ == "__main__":
    # 加载模板
    template = cv2.imread('images/10.JPG',1)
    # 初始化
    am = C_ammerter(template)
    # 运行
    am.am_run()
    # 结束
    am.close()


