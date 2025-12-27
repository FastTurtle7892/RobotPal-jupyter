import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import PIL.Image
import numpy as np
import ipywidgets as widgets
from IPython.display import display
import traitlets
from jetbot import Robot, Camera, bgr8_to_jpeg
from SCSCtrl import TTLServo
import threading
import time
import os
from datetime import datetime

class WorkingAreaFind(threading.Thread):
    def __init__(self, robot, camera, widgets_dict, camera_center=(112, 112)):
        super().__init__()
        self.robot = robot
        self.camera = camera
        self.w = widgets_dict  # 메인에서 전달받은 위젯 딕셔너리
        self.center_x, self.center_y = camera_center

        self.th_flag=True
        self.flag = 1
        
        self.imageInput = 0
        
        #area 찾는 순서 변수
        
        self.w['flaglbl'].value = str(self.flag)
        
    def run(self):
        while self.th_flag:
            self.imageInput = self.camera.value
            #BGR to HSV
            hsv = cv2.cvtColor(self.imageInput, cv2.COLOR_BGR2HSV)
            #blur
            hsv = cv2.blur(hsv, (15, 15))
                        
            #areaA, areaB Color searching
            areaA_mask = cv2.inRange(hsv, self.colors['areaA']['lower'], self.colors['areaA']['upper'])
            areaA_mask = cv2.erode(areaA_mask, None, iterations=2)
            areaA_mask = cv2.dilate(areaA_mask, None, iterations=2)

            areaB_mask = cv2.inRange(hsv, self.colors['areaB']['lower'], self.colors['areaB']['upper'])
            areaB_mask = cv2.erode(areaB_mask, None, iterations=2)
            areaB_mask = cv2.dilate(areaB_mask, None, iterations=2)

            # 해당 영역에 대한 윤곽선 따기
            AContours, _ = cv2.findContours(areaA_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            BContours, _ = cv2.findContours(areaB_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #A영역 or B영역을 찾았다면
            if AContours and self.flag == 1:
                self.findCenter("areaA", AContours)
            
            elif BContours and self.flag == 2:
                self.findCenter("areaB", BContours)
            
            #두 영역 모두 못찾았다면, 찾아가는 중이다.
            else:
                cv2.putText(self.imageInput, "Finding...", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
                self.w['image_widget'].value = bgr8_to_jpeg(self.imageInput)
            time.sleep(0.1)
    
    #name : A, B 구분용도, Contours 각 영역의 윤곽선 값
    def findCenter(self, name, Contours):  
        c = max(Contours, key=cv2.contourArea)
        ((box_x, box_y), radius) = cv2.minEnclosingCircle(c)

        X = int(box_x)
        Y = int(box_y)
        
        error_Y = abs(self.center_y - Y)
        error_X = abs(self.center_x - X)
        
        if error_Y < 15 and error_X < 15:
            #A영역이 가까이 오게됨
            if name == "areaA" and self.flag == 1:
                self.flag = 2
                findArea = "areaB"
                self.w['goallbl'].value = findArea
                
                self.w['areaAlbl'].value = "areaA" + " Goal!"
                self.w['flaglbl'].value = str(self.flag)
                
            #B영역이 가까이 오게됨
            elif name == "areaB" and self.flag == 2:              
                self.flag = 1       
                findArea = "areaA"
                self.w['goallbl'].value = findArea

                self.w['areaBlbl'].value = "areaB" + " Goal!"
                self.w['flaglbl'].value = str(self.flag)
            # ocr 쓰레드 실행 
                
        self.w['image_widget'].value = bgr8_to_jpeg(self.imageInput)
        
    def stop(self):
        self.th_flag = False
        self.robot.stop()