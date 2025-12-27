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

class RobotMoving(threading.Thread):
    def __init__(self, robot, camera, model, device, widgets_dict):
        super().__init__()

        self.robot = robot
        self.camera = camera
        self.model = model
        self.device = device
        self.w = widgets_dict

        self.th_flag = True
        self.angle = 0.0
        self.angle_last = 0.0
        road_following_active = False

        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

    def preprocess(self, image):
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(self.device).half()
        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return image[None, ...]
        
    def run(self):

        global road_following_active

        while self.th_flag:
            # 만약 전역 변수를 안 쓴다면 이 부분은 생략 가능합니다.
            if not road_following_active:
                self.robot.stop()
                time.sleep(0.1)
                continue

            image = self.camera.value
            xy = self.model(self.preprocess(image)).detach().float().cpu().numpy().flatten()
            x = xy[0]
            y = (0.5 - xy[1]) / 2.0
            
            self.w['x_slider'].value = x
            self.w['y_slider'].value = y
            self.w['speed_slider'].value = self.w['speed_gain_slider'].value
            self.w['image_widget'].value = bgr8_to_jpeg(image)
            
            #인공지능 무인운반로봇(AGV)의 속도 표시
            self.w['speed_slider'].value = self.w['speed_gain_slider'].value
            
            self.w['image_widget'].value = bgr8_to_jpeg(image)
            
            #조향값 계산
            self.angle = np.arctan2(x, y)
            
            if not self.th_flag:
                break
            #PID 제어를 이용한 모터 제어
            pid = self.angle * self.w['steering_gain_slider'].value + (self.angle - self.angle_last) * self.w['steering_dgain_slider'].value
            self.angle_last = self.angle

            #슬라이더에 표시
            self.w['steering_slider'].value = pid + self.w['steering_bias_slider'].value
            self.robot.left_motor.value = max(min(self.w['speed_slider'].value + self.w['steering_slider'].value, 1.0), 0.0)
            self.robot.right_motor.value = max(min(self.w['speed_slider'].value - self.w['steering_slider'].value, 1.0), 0.0)
            time.sleep(0.05)
        self.robot.stop()
    


    def stop(self):
        self.th_flag = False
        self.robot.stop()