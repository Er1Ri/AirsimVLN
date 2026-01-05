# -*- coding: utf-8 -*-
# @Time    : 2023/12/23  22:24
# @Author  : mariswang@rflysim
# @File    : tello_wrapper.py
# @Software: PyCharm
# @Describe: 
# -*- encoding:utf-8 -*-


import sys
import threading
import time


from djitellopy import Tello
import math
import numpy as np
import io
import cv2

import random
import torch
from torchvision.ops import box_convert
from PIL import Image

from ram.models import ram
from ram import inference_ram as inference
from ram import get_transform
from groundingdino.util.inference import load_model, load_image, predict, annotate

from transformers import pipeline

BOX_TRESHOLD = 0.25
TEXT_TRESHOLD = 0.25



class TelloWrapper:
    def __init__(self):
        self.client = Tello()
        self.client.connect()


        #启动视频流
        self.client.streamon()  # 开启视频传输
        t = threading.Thread(target=self.get_stream)
        t.setDaemon(True)
        t.start()

        self.head_img = None #tello 前置摄像头


         #目标识别
        ram_model = ram(pretrained='../../pretrained/ram_swin_large_14m.pth',
                         image_size=384,
                         vit='swin_l')
        ram_model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        self.device = device
        self.ram_model = ram_model.to(device)
    
        #目标检测
        self.dino_model = load_model("../../config/GroundingDINO_SwinT_OGC.py", "../../pretrained/groundingdino_swint_ogc.pth")

        #深度视觉估算
        checkpoint = "vinvino02/glpn-nyu"
        self.depth_estimator = pipeline("depth-estimation", model=checkpoint)




    def get_stream(self):
        while True:
            self.head_img = self.client.get_frame_read().frame
            time.sleep(0.01)

    def takeoff(self):
        self.client.takeoff()

    def land(self):
        self.client.land()


    def turn_left(self,degree=10):
        """
        左转degree度
        :return:
        """
        degree = 10*degree #tello base is 0.1度
        self.client.rotate_counter_clockwise(degree)


    def turn_right(self,degree=10):
        """
        右转degree度
        :return:
        """
        degree = 10*degree #tello base is 0.1度
        self.client.rotate_clockwise(degree)


    def forward(self, distance):
        """
        向前移动, 太少了不动
        distance: 距离，米
        :return:
        """
        distance = distance*100 # tello base is 1cm
        self.client.move_forward(distance) #向前移动50cm


    def back(self, distance):
        """
        向后移动, 太少了不动
        distance: 距离，米
        :return:
        """
        distance = distance*100 # tello base is 1cm
        self.client.move_back(distance) #向前移动50cm


    def up(self, distance):
        """
        向上移动, 太少了不动
        distance: 距离，米
        :return:
        """
        distance = distance*100 # tello base is 1cm
        self.client.move_up(distance) #向前移动50cm


    def down(self, distance):
        """
        向下移动, 太少了不动
        distance: 距离，米
        :return:
        """
        distance = distance*100 # tello base is 1cm
        self.client.move_down(distance) #向前移动50cm
        
        
    def get_image(self):
        """
        获得前置摄像头渲染图像
        :return:
        """
        return self.head_img

    def get_drone_state(self):
        """
        获得无人机状态,
        :return:{'pitch': int, 'roll': int, 'yaw': int}
        """
        return self.client.query_attitude()


    def get_depth_estimator(self, img):
        """
        在图像 img 上运行深度视觉预估
        :param img:cv2的图片
        :return:predicted_depth # 图片上像素点距离无人机的距离
        """
        img_pil = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        predictions = depth_estimator(img_pil)

        return predictions["predicted_depth"]
        
    
    def get_objects(self, img):
        """
        在图像 img 上运行对象检测模型
        :param img:
        :return:obj_list # 图片上的目标名称列表
        """
        #目标检测
        #opencv图片转bytes,pil可直接读取
        imgbytes = cv2.imencode(".jpg", img)[1].tobytes()
        byte_stream = io.BytesIO(imgbytes)
        
        transform = get_transform(image_size=384)
        image = transform(Image.open(byte_stream)).unsqueeze(0).to(self.device)
        res = inference(image, self.ram_model)
        obj_name_list = res[0].split(" | ") #res[0] 英文，res[1]中文
        return obj_name_list
    
    def ob_objects(self,obj_name_list):
        """
        注意需要先执行get_image，
        在图像 img 上运行对象检测模型，获得目标列表 [ <对象名称、距离、角度（以度为单位）>,...]
        :return:对象名称列表、对象信息列表、bbox图
        """

        TEXT_PROMPT = " | ".join(obj_name_list)
        #目标检测
        imgbytes = cv2.imencode(".jpg", self.head_img)[1].tobytes()
        byte_stream = io.BytesIO(imgbytes)
        
    
        image_source, image = load_image(byte_stream)
        
        boxes, logits, phrases = predict(
            model=self.dino_model,
            image=image, 
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )

        h, w, _ = image_source.shape
        boxes_unnorm = boxes * torch.Tensor([w, h, w, h])
        boxes_xyxy = box_convert(boxes=boxes_unnorm, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        
        #[xmin, ymin, xmax, ymax]
        obj_locs = boxes_xyxy

        #深度预测
        img_camera_distance = self.get_depth_estimator(self.head_img)  #相机距离

        
        final_obj_list = [] #最终结果列表
        #构建目标结果
        index = 0
        for bbox in obj_locs:
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)

            camera_distance = img_depth_perspective[center_y, center_x] #相机距离

            obj_name =  phrases[index]#获得目标名称，可能有多个

            obj_info = (obj_name, camera_distance, center_x, center_y)
            final_obj_list.append(obj_info)
            index = index + 1

        #画框
        #annotated_frame：cv2的图片，image_source：pil图片
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        return phrases, final_obj_list, annotated_frame


    def ob_objects_llm(self,obj_name_list):
        """
        注意需要先执行get_image，为llm提供观测结果
        在图像 img 上运行对象检测模型，获得目标列表 [ <对象名称、距离、角度（以度为单位）>,...] , 给到llm用于推理
        :return:[ <对象名称、距离、角度（以度为单位）>,...] 如[(门，0.53，22)，(椅子，4.84，-21)]
        """
        #获得识别结果
        ob_list, final_obj_list, annotated_frame = self.ob_objects(obj_name_list)

        final_result = []

        for obj_info in final_obj_list:
            item = (obj_info[0], obj_info[1], obj_info[3]) #obj_name, camera_distance, angel_degree
            final_result.append(item)

        return final_result








