from system_core_yolov2 import darknet
import os
import cv2
import numpy as np
import time
from sklearn.metrics import precision_recall_fscore_support

netMain = None
metaMain = None
altNames = None
netMain2 = None
metaMain2 = None
altNames2 = None

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def YOLO():

    # forward recognition

    global metaMain, netMain, altNames

    configPath = "./yolov2_blister_recog.cfg"
    weightPath = "./yolov2_blister_recog_final.weights"
    metaPath = "./blister_recog.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
###############################################################################

    gt = []
    cls = -1
    # img_dir = "./yolo_back/JPEGImages"
    # img_num = len(os.listdir(img_dir))
    # print(img_num)
    # img_pre_cls = img_num/230
    cap0 = cv2.VideoCapture(1)
    cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # image input
    pred = []

    while(True):
        # backward recognition
        # print(im)
        ret0, frame0 = cap0.read()
        assert ret0
        # back_img_path = "{}/{}.jpg".format(img_dir, 5749)
        cfgPath = "./yolov2_blister_recog.cfg"
        back_wegPath = "./yolov2_blister_recog_final.weights"
        mtPath = "./blister_recog.data"
        # img = cv2.imread(back_img_path)
        # print("Starting the YOLO loop...")
        darknet_image = darknet.make_image(darknet.network_width(netMain),
                                           darknet.network_height(netMain), 3)
        frame_rgb = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
        detections, prob = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)

        # back_detections, back_prob = darknet.back_performDetect(imagePath=frame0, thresh=0.25, configPath=cfgPath
        #                                                    , weightPath=back_wegPath, metaPath=mtPath, showImage=False)

        pred_img = frame0
        # print(len(detections))
        if len(detections) != 0:
            pred_class = detections[0][0]
            pred_confid = detections[0][1]
            pred_bbox = detections[0][2]
            x, y, w, h = pred_bbox[0], \
                         pred_bbox[1], \
                         pred_bbox[2], \
                         pred_bbox[3]
            xmin, ymin, xmax, ymax = convertBack(
                float(x), float(y), float(w), float(h))
            # avoid < 0 and > frame.shape
            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > pred_img.shape[1]:
                xmax = pred_img.shape[1]
            if ymax > pred_img.shape[0]:
                ymax = pred_img.shape[0]
            # print(pred_bbox)
            pred_class = str(pred_class).split('\'')[1]
            print(pred_class)
        else:
            pred_class = ''
            pred_confid = 0
            xmin = 0
            ymin = 0
            xmax = 0
            ymax = 0

        cv2.rectangle(pred_img, (int(xmin*1.73),int(ymin*1.15)),(int(xmax*1.73),int(ymax*1.15)), (0, 0, 255),2)
        blank = np.ones((170, 640, 3), np.uint8)
        pred_show = cv2.vconcat((pred_img, blank))
        cv2.putText(pred_show, text="Blister class : ", org=(0, 520), fontFace=1, fontScale=2, thickness=2, color=(255, 255, 255))
        cv2.putText(pred_show, str(pred_class), org=(0, 555), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.7, color=(0, 0, 255), thickness=2)
        cv2.putText(pred_show, text="confidence score : {:.2f}%".format(pred_confid*100), org=(0, 590), fontFace=1, fontScale=2, thickness=2, color=(255, 255, 255))
        cv2.imshow('Pred_show', pred_show)
        key = cv2.waitKey(30)& 0xFF
        if(key == 27):
            cv2.destroyAllWindows()
            break


    # print(back_detections)
    # print(for_detections)

if __name__ == "__main__":
    YOLO()