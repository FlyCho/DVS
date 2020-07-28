#!/usr/bin/python
# -*- coding: UTF-8 -*-
import json
import pyzbar.pyzbar as pyzbar
import cv2
import numpy as np
import re
import time
import tkinter as tk
import threading
from PIL import Image, ImageTk
import websocket
import _thread as thread
def barcode_decode(image, ws):
    barcodes = pyzbar.decode(image)           # 解碼barcode
    for barcode in barcodes:
        (x, y, w, h) = barcode.rect           # 畫出barcode邊框位子
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        barcodeData = barcode.data.decode('utf-8')
        barcodeType = barcode.type

        print(barcodeData)

        text = "{} ({})".format(barcodeData, barcodeType)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (0, 0, 125), 2)
        data_list = []

        if barcodeType == "QRCODE":
            # print("Barcode data"+barcodeData)
            for i in re.split(r'(;+)', barcodeData):
                if ";" not in i:
                    data_list.append(i)
            print('datalist'+str(data_list))
            data_list = data_list[1:-1]
            if len(data_list) > 0 :
                medicine_name = medicine_dict[data_list[5]]
                medicine_photo = medicine_photo_dict[medicine_name]
                ws.send(json.dumps({"type":"qrcode","id":"ticket", "name":data_list[0], "subject":data_list[1], "date":data_list[2], "sum":data_list[-1],"doctor":data_list[4], "medicineID":data_list[5], "medicineName":medicine_name, "medicinePhoto":medicine_photo}))

            # ws.send(json.dumps(medicine_photo_dict))
            # ws.send(json.dumps(medicine_dict))
                time.sleep(1)
        elif barcodeType == "CODE39":
            try:
                blisterbag_dict[barcodeData]["type"] = "barcode"
                blisterbag_dict[barcodeData]["bagid"] = barcodeData

                print(blisterbag_dict[barcodeData])
                ws.send(json.dumps(blisterbag_dict[barcodeData]))
                time.sleep(1)
            except KeyError:
                print("continue")


    return image

def on_message(ws, message):
    print(str(message))

def on_error(ws, error):
    print(error)

def on_close(ws):
    print("### closed ###")

def on_open(ws):
    def run(*args):
        cap = cv2.VideoCapture(2)
        while (True):
            ws.on_open = on_open
            ret, frame = cap.read()
            assert ret
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image = barcode_decode(gray, ws)
            cv2.waitKey(5)
            cv2.imshow("camera", image)
        cap.release()
        cv2.destroyAllWindows()
        ws.close()
        print("thread terminating...")
    thread.start_new_thread(run, ())

# 偵測條碼方法
def detect():
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp("ws://localhost:9001",
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()

    # ws.run_forever()

# 讀取藥品代碼、名稱、圖片代號
def medicine():
    medicine_dict = {}
    medicine_photo = {}
    blisterbag = {}
    with open('drugsA_code.txt') as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split(",")
            medicine_dict[line[0]] = line[1]
    with open('blister_names.txt') as f:
        blister_index = 1
        for blister in f.readlines():
            blister = blister.strip('\n')
            medicine_photo[blister] = str(blister_index)
            blister_index += 1

    with open('blisterbag.txt',encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split(",")
            blister_item = {}
            medicine_index = 1
            for blister in line:
                if ":" in blister:
                    blister = blister.split(":")
                    blister_item[medicine_index] = {}
                    blister_item[medicine_index]["id"] = blister[0]
                    blister_item[medicine_index]["name"] = medicine_dict[blister[0]]
                    blister_item[medicine_index]["sum"] = blister[1]
                    medicine_index += 1

            blisterbag[line[0]] = {}
            blisterbag[line[0]]["name"] = line[1]
            blisterbag[line[0]]["subject"] = line[2]
            blisterbag[line[0]]["doctor"] = line[3]
            blisterbag[line[0]]["blister"] = blister_item
            blisterbag[line[0]]["date"] = line[-1]
    return medicine_dict, medicine_photo, blisterbag


def check(canvas,windows):

    test = canvas.create_text(1000, 310, text="檢核結果123456", fill='black', font="Times 30 bold")
    print("check check")

def main():
    detect()
    #data_list = ['', '', '', '', '', '', '', '', '', '']
    # data_list = ['林明昆', '13', '1070814', '14', '張經緯', '22440', '1', 'BID', 'PO', '28']
    #gui(data_list)

if __name__ == '__main__':
    medicine_dict, medicine_photo_dict, blisterbag_dict = medicine()  # 載入藥品代碼與藥名
    print(blisterbag_dict)
    main()