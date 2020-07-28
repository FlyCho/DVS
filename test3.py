# import pyzbar.pyzbar as pyzbar
# import cv2
# import numpy as np
# import re
# import time
# import tkinter as tk
#
#
# def barcode_decode(image):
#     barcodes = pyzbar.decode(image)           # 解碼barcode
#     for barcode in barcodes:
#         (x, y, w, h) = barcode.rect           # 畫出barcode邊框位子
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
#
#         barcodeData = barcode.data.decode('utf-8')
#         barcodeType = barcode.type
#
#         text = "{} ({})".format(barcodeData, barcodeType)
#         cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
#                     .5, (0, 0, 125), 2)
#         if(barcodeType == "CODE39"):
#             print("barcode")
#         elif(barcodeType == "QRCODE"):
#             print("QRcode")
#
#     return image
#
# # 偵測條碼方法
# def detect():
#     cap = cv2.VideoCapture(3)
#     while(True):
#         ret, frame = cap.read()
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         # image = barcode_decode(gray)  # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         image = barcode_decode(gray)
#         cv2.waitKey(5)
#         cv2.imshow("camera", image) # cv2.imshow("camera", image)
#     cap.release()
#     cv2.destroyAllWindows()
#
# def check(canvas):
#     test = canvas.create_rectangle(50, 50, 150, 150, outline="black", width=4)
#     canvas.pack
#
# def main():
#     detect()
#
#
# if __name__ == '__main__':
#     main()
medicine_dict = {}
with open('drugsA_code.txt') as f:
    for line in f.readlines():
        line = line.strip('\n')
        line = line.split(",")
        medicine_dict[line[0]] = line[1]

blisterbag = {}
with open('blisterbag.txt') as f:
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
    print(blisterbag["000014383177A"]["blister"][1])