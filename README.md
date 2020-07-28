# DVS
Dispensation Verification System
### Demo
+ [Demo](https://youtu.be/J1P_9GbXVAo)
### Introduction
Dispensation Verification System is implemented by blister identification system.

There are two blister identification systems : YOLOv2(single side) & Fast ROR(both side)

### Install
+ Python3.6
+ tensorflow 1.12.0
+ openCV
### Step 1 : Open the barcode server
```
python barcode_server_9001.py
```
### Step 2 : Open the system core server
```
python system_core_server_9002.py
```
### Step 3 : Open the barcode client
```
python barcode_client_9001.py
```
### Step 4 : Open the system core client
```
Fast ROR

python system_core_FastROR/system_core_FastROR_client_9002.py
```
```
YOLOv2

python system_core_yolov2/system_core_yolov2_client_9002.py
```
