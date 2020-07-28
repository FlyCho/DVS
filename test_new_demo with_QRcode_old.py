import tensorflow.contrib.slim as slim
import tensorflow as tf
import cv2
import time
import numpy as np
from model import pix2pix
from IFPA_utils.network import PIL
from pix2pix_args import pix2pix_args
from IPAF_args import IFPA_args
from demo_img_preprocess import img_preprocess
import os
import json
import websocket
import _thread as thread
def on_message(ws, message):
    if(str(message) == "no blister"):
        print("*******************************************************************************************")
        print("-------------------------------------------------------------------------------------------")
    print(str(message))

def on_error(ws, error):
    print(error)

def on_close(ws):
    print("### closed ###")

def on_open(ws):
    def run(*args):
        recog_num = 0                      # estimate 10 frame
        recog_none = 0
        recog_list = []                    # the list of storing predict blister
        while (True):

            # camera input
            # =======================================================================
            ret0, frame0 = cap0.read()
            assert ret0
            ret1, frame1 = cap1.read()
            assert ret1

            # with attention
            # =======================================================================
            # ori_img = np.concatenate((frame0, frame1), axis=1)
            # cv2.imshow('ori', ori_img)
            # if cv2.waitKey(1) & 0xFF == ord('b'):
            #     cv2.imwrite("{}/{}.jpg".format(BG_image_path, "BG_L"), frame0)
            #     cv2.imwrite("{}/{}.jpg".format(BG_image_path, "BG_R"), frame1)
            #     BG_flag = 1
            #     # BG_L = frame1
            #     # BG_R = frame2
            #     print("BackGround image save sucessful!")
            #     pass
            # if BG_flag == 0:
            #     print("Background pic is not ready!")
            #     time.sleep(1)
            # else:
            #     BG_L = cv2.imread("{}/{}.jpg".format(BG_image_path, "BG_L"))
            #     BG_R = cv2.imread("{}/{}.jpg".format(BG_image_path, "BG_R"))
            #     subres_L = cv2.subtract(frame0, BG_L)
            #     subres_R = cv2.subtract(frame1, BG_R)
            #     sub_img = np.concatenate((subres_L, subres_R), axis=1)
            #     cv2.imshow("sub", sub_img)
            #     # cv2.waitKey(0)
            #     img1 = attation(subres_L, frame0)
            #     # cv2.imshow("1", res_img_L)
            #     img2 = attation(subres_R, frame1)
            #
            #     img1 = cv2.resize(img1, (256, 256), cv2.INTER_AREA)
            #     img2 = cv2.resize(img2, (256, 256), cv2.INTER_AREA)
            #========================================================================
            img1 = cv2.resize(frame0, (256, 256), cv2.INTER_AREA)
            img2 = cv2.resize(frame1, (256, 256), cv2.INTER_AREA)


            # img1 = cv2.subtract(img1, frame0_bg)
            # img2 = cv2.subtract(img2, frame1_bg)
            # =======================================================================

            # image read
            # =======================================================================

            # img1 = cv2.imread("{}/img_{}.jpg".format(input_path, "23"))
            # img2 = cv2.imread("{}/img_{}.jpg".format(input_path, "24"))
            # img1 = cv2.resize(img1, (256, 256), cv2.INTER_AREA)
            # img2 = cv2.resize(img2, (256, 256), cv2.INTER_AREA)
            # =======================================================================
            result1 = np.concatenate((img, img1), axis=1)
            result2 = np.concatenate((img, img2), axis=1)




            real = np.asarray(pix2pix_model.demo(pix2pix_args, result1, result2, False)).reshape(-1, 1)

            # print(real[0])
            # print(real[1])
            if real[0] > real_threshold or real[1] > real_threshold:
                img1_pre = cv2.imread("{}/test_0001.png".format(input_path))
                img2_pre = cv2.imread("{}/test_0002.png".format(input_path))
            else:
                img1_pre = np.zeros((256, 256, 3), np.uint8)
                img2_pre = np.zeros((256, 256, 3), np.uint8)

            # if real[1] > real_threshold:
            #     img2_pre = cv2.imread("{}/test_0002.png".format(input_path))
            # else:
            #     img2_pre = np.zeros((256, 256, 3), np.uint8)

            img1_cal, img1_p, img1_c, mass1 = img_preprocess(img1_pre, img1, frame0, M_resize2ori)
            img2_cal, img2_p, img2_c, mass2 = img_preprocess(img2_pre, img2, frame1, M_resize2ori)

            img2_cal = cv2.warpPerspective(img2_cal, M_flip, (224, 448))

            result3 = np.concatenate((img1_pre, img2_pre), axis=1)
            result4 = np.concatenate((img1_cal, img2_cal), axis=1)

            result5 = np.concatenate((img1_p, img2_p), axis=1)
            result6 = np.concatenate((img1_c, img2_c), axis=1)

            if real[0] > real_threshold or real[1] > real_threshold:

                RTT_img = cv2.resize(result4, (224, 224), cv2.INTER_AREA)
                RTT_img = cv2.cvtColor(RTT_img, cv2.COLOR_BGR2RGB)
                RTT_img = RTT_img.reshape(1, 224, 224, 3)

                pred = sess1.run([logits_list], feed_dict={
                    x: RTT_img,
                    training_flag: False
                })

                softmax_x = np.asarray(pred).reshape(-1).tolist()
                softmax_x = softmax(softmax_x)
                softmax_x = softmax_x.reshape(-1, 1)

                # ten times recognition


                RTT_pred = np.argmax(pred[0][0], 1)

                if(recog_num==15):
                    # print(recog_list)
                    time_dict = {}
                    for i in recog_list:
                        if i in time_dict:
                            time_dict[i] += 1
                        else:
                            time_dict[i] = 1
                    # print(time_dict)
                    time_dict_sort = sorted(time_dict.items(), key=lambda d: d[1], reverse=True)
                    # print(blister_names[time_dict_sort[0][0]])
                    if len(time_dict_sort)>0:
                        ws.send(json.dumps({"blister":blister_names[time_dict_sort[0][0]]}))


                    recog_list = []
                    recog_num = 0

                recog_list.append(RTT_pred[0])
                recog_num += 1

                RTT_pred_confidence = softmax_x[RTT_pred][0][0]
                RTT_pred_name = "{}% {}".format(int(RTT_pred_confidence * 100), blister_names[RTT_pred[0]])
            else:
                RTT_pred_name = "None"
                recog_none += 1
                if recog_none > 9:
                    recog_list = []
                    recog_none = 0

            result7 = np.concatenate((result6, result3))
            result7 = np.concatenate((result7, result5))
            blank = np.zeros((448, 320, 3), np.uint8)
            result4 = np.concatenate((result4, blank), axis=1)
            blank = np.zeros((320, 768, 3), np.uint8)
            result4 = np.concatenate((result4, blank), axis=0)
            result8 = np.concatenate((result7, result4), axis=1)

            cv2.rectangle(result8, pt1=(500, 550), pt2=(1300, 615), color=(255, 255, 255), thickness=-1)
            cv2.putText(result8, text="Blister class : ", org=(500, 575), fontFace=1, fontScale=2, thickness=2,
                        color=(0, 0, 0))
            cv2.putText(result8, RTT_pred_name, (500, 610), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(result8, pt1 = (0,0), pt2 = (200,30),color = (255,255,255),thickness=-1)
            cv2.putText(result8, text="Image Input",org=(0,25), fontFace=1, fontScale=2,thickness=2,color=(0,0,0))
            cv2.rectangle(result8, pt1=(0, 256), pt2=(310, 286), color=(255, 255, 255), thickness=-1)
            cv2.putText(result8, text="Pixel2Pixel Output", org=(0, 281), fontFace=1, fontScale=2, thickness=2, color=(0, 0, 0))
            cv2.rectangle(result8, pt1=(0, 512), pt2=(480, 542), color=(255, 255, 255), thickness=-1)
            cv2.putText(result8, text="After Image Pre-processing", org=(0, 537), fontFace=1, fontScale=2, thickness=2,
                        color=(0, 0, 0))
            cv2.rectangle(result8, pt1=(512, 448), pt2=(960, 476), color=(255, 255, 255), thickness=-1)
            cv2.putText(result8, text="RTT Image", org=(650, 473), fontFace=1, fontScale=2, thickness=2,
                        color=(0, 0, 0))

            cv2.imshow("1", result8)

            key = cv2.waitKey(30) & 0xFF
            if (key == 27):
                cap0.release()
                cap1.release()
                cv2.destroyAllWindows()
                break
        ws.close()
        print("thread terminating...")
    thread.start_new_thread(run, ())


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def attation(sub_img, ori_img):
    # 轉HSV
    imgHSV = cv2.cvtColor(sub_img, cv2.COLOR_BGR2HSV)
    # mask 去背
    lower_green = np.array([0, 0, 0])
    upper_green = np.array([180, 255, 40])
    mask = cv2.inRange(imgHSV, lower_green, upper_green)
    # 圖片擴張
    dilate = cv2.dilate(mask, None, iterations=1)
    # 圖片侵蝕
    erode = cv2.erode(dilate, None, iterations=15)
    # ori = 3
    n_mask = cv2.cvtColor(erode, cv2.COLOR_GRAY2BGR)
    not_erode = cv2.bitwise_not(erode)

    # 森德的演算法
    ret, mask_1 = cv2.threshold(not_erode, 240, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(mask_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_area = 0
    index = 0
    for c in range(len(contours)):
        area = cv2.contourArea(contours[c])
        if (area > max_area):
            max_area = area
            index = c
    blank = np.zeros(n_mask.shape, np.uint8)
    rect_blank = np.zeros(n_mask.shape, np.uint8)

    if index == 0:
        crop = ori_img
    else :
        hull = cv2.convexHull(contours[index])
        cv2.drawContours(blank, [hull], 0, (255, 255, 255), cv2.FILLED, cv2.LINE_AA)
        rect = cv2.minAreaRect(hull)
        box = box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(rect_blank, [box], 0, (255, 255, 255), cv2.FILLED, cv2.LINE_AA)
        # =================================================================================

        # 因藥排有些會被切到 所以再擴張一次
        blank = cv2.dilate(blank, None, iterations=5)
        n_mask = cv2.cvtColor(not_erode, cv2.COLOR_GRAY2BGR)

        # 擷取手持藥排影像
        crop = cv2.bitwise_and(ori_img, blank)

    return crop

blister_pack_names_input_path = "./blister_names.txt"
blister_names = []
f = open(blister_pack_names_input_path, 'r')
for l in f:
    l = l.strip('\n')
    blister_names.append(l)
f.close()

g1 = tf.Graph()
g2 = tf.Graph()

with g1.as_default():
    IFPA_args = IFPA_args
    class_num = 230

    x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])  # input image size
    training_flag = tf.placeholder(tf.bool)
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    local_x_list, logits_list = PIL(x, training=training_flag, class_num=class_num, drop_rate=IFPA_args.drop_rate,
                                    reduction_ratio=IFPA_args.reduction_ratio,
                                    part=IFPA_args.part, straight=IFPA_args.straight, half=IFPA_args.half,
                                    all=IFPA_args.all).model

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    variables = slim.get_variables_to_restore()

with g2.as_default():
    pix2pix_args = pix2pix_args

with tf.Session(graph=g1, config=tf.ConfigProto(allow_soft_placement=True)) as sess1:
    with tf.Session(graph=g2, config=tf.ConfigProto(allow_soft_placement=True)) as sess2:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        saver.restore(sess1, './checkpoint/IFPA/model.ckpt-58')
        pix2pix_model = pix2pix(sess2, image_size=pix2pix_args.fine_size, batch_size=pix2pix_args.batch_size,
                                output_size=pix2pix_args.fine_size, dataset_name=pix2pix_args.dataset_name,
                                checkpoint_dir=pix2pix_args.checkpoint_dir, sample_dir=pix2pix_args.sample_dir)
        pix2pix_model.demo(pix2pix_args, 0, 0, True)

        cap0 = cv2.VideoCapture(0)
        cap1 = cv2.VideoCapture(1)
        cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        input_path = "./test"
        src = np.array([[224, 448], [0, 448], [0, 0], [224, 0]], np.float32)
        dst = np.array([[0, 0], [224, 0], [224, 448], [0, 448]], np.float32)
        M_flip = cv2.getPerspectiveTransform(src, dst)

        ori_frame_size = np.array([[0, 0], [640, 0], [640, 480], [0, 480]], np.float32)
        resize_frame_size = np.array([[0, 0], [256, 0], [256, 256], [0, 256]], np.float32)
        M_resize2ori = cv2.getPerspectiveTransform(resize_frame_size, ori_frame_size)

        img = np.zeros((256, 256, 3), np.uint8)

        # ============ Discriminator Threshold =============================
        real_threshold = 0.64
        # ==================================================================

        count = 1
        global frame0_bg, frame1_bg
        BG_image_path = "./Background"
        BG_flag = 0
        if not os.path.exists(BG_image_path):
            os.makedirs(BG_image_path)

        # while(True):
        #     ret0, frame0 = cap0.read()
        #     assert ret0
        #     ret1, frame1 = cap1.read()
        #     assert ret1
        #     if (count > 10):
        #         frame0_bg = cv2.resize(frame0, (256, 256), cv2.INTER_AREA)
        #         frame1_bg = cv2.resize(frame1, (256, 256), cv2.INTER_AREA)
        #         break
        #     count += 1

        websocket.enableTrace(True)
        ws = websocket.WebSocketApp("ws://localhost:9002",
                                    on_message=on_message,
                                    on_error=on_error,
                                    on_close=on_close)
        ws.on_open = on_open
        ws.run_forever()

        coord.request_stop()
        coord.join(threads)
