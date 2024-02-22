#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import torch
import rospy
import numpy as np
from ultralytics import YOLO
from time import time

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from yolov8_ros_msgs.msg import BoundingBox, BoundingBoxes


class Yolo_Dect:
    def __init__(self):

        # load parameters
        weight_path = rospy.get_param('~weight_path', '')
        image_topic = rospy.get_param(
            '~image_topic', '/camera/color/image_raw')
        pub_topic = rospy.get_param('~pub_topic', '/yolov8/BoundingBoxes')
        self.camera_frame = rospy.get_param('~camera_frame', '')
        conf = rospy.get_param('~conf', '0.5')
        self.visualize = rospy.get_param('~visualize', 'True')

        self.model = YOLO(weight_path)
        #self.model.fuse()

        self.model.conf = conf
        self.color_image = Image()
        self.getImageStatus = False

        # Load class color
        self.classes_colors = {}

        # image subscribe
        self.color_sub = rospy.Subscriber(image_topic, Image, self.image_callback,
                                          queue_size=1, buff_size=52428800)

        # output publishers
        self.position_pub = rospy.Publisher(
            pub_topic,  BoundingBoxes, queue_size=1)

        self.image_pub = rospy.Publisher(
            '/yolov8/detection_image',  Image, queue_size=1)

        # if no image messages
        while (not self.getImageStatus):
            rospy.loginfo("waiting for image.")
            rospy.sleep(1)

    def image_callback(self, image):

        self.boundingBoxes = BoundingBoxes()
        self.boundingBoxes.header = image.header
        self.boundingBoxes.image_header = image.header
        self.getImageStatus = True
        self.color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(
            image.height, image.width, -1)

        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)

        results = self.model(self.color_image, show=False, conf=0.3, classes=[0, 2, 5, 7])

        self.dectshow(results, image.height, image.width)

        cv2.waitKey(1)

    def normalize(self, imgxyxy, shape_data, shape_detect):
        '''
            imgxyxy : detect xyxy
            shape_data : data
            shape_detect : detect
        '''
        ratio = max([shape_data[0]/shape_detect[0], shape_data[1]/shape_detect[1]])
        for i, det in enumerate(imgxyxy):
            imgxyxy[i] = imgxyxy[i]*ratio
        resizey = ratio * shape_detect[0]  # resize shape of detect to data(Y)
        resizex = ratio * shape_detect[1]  # resize shape of detect to data(X)
        if resizey != shape_data[0]:  # delete additional padding(Y)
            paddingy = (shape_data[0]-resizey) / 2
            imgxyxy[1] = imgxyxy[1] + paddingy
            imgxyxy[3] = imgxyxy[3] + paddingy
        elif resizex != shape_data[1]:    # delete additional padding(X)
            paddingx = (shape_data[1] - resizex) / 2
            imgxyxy[0] = imgxyxy[0] + paddingx
            imgxyxy[2] = imgxyxy[2] + paddingx
        return imgxyxy
        
    def dectshow(self, results, height, width):

        # self.frame = results[0].plot()
        print(str(results[0].speed['inference']))
        # fps = 1000.0/ results[0].speed['inference']
        # cv2.putText(self.frame, f'FPS: {int(fps)}', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        for result in results[0].boxes:
            boundingBox = BoundingBox()
            boundingBox.xmin= xmin = np.int64(result.xyxy[0][0].item())
            boundingBox.ymin= ymin = np.int64(result.xyxy[0][1].item())
            boundingBox.xmax= xmax = np.int64(result.xyxy[0][2].item())
            boundingBox.ymax= ymax = np.int64(result.xyxy[0][3].item())
            self.xyxy = self.normalize([xmin, ymin, xmax, ymax], self.color_image.shape, [384,640])
            boundingBox.Class = results[0].names[result.cls.item()]
            boundingBox.probability = result.conf.item()
            self.boundingBoxes.bounding_boxes.append(boundingBox)
            
            if boundingBox.Class == 'person':
              cv2.rectangle(self.color_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 4)
              cv2.putText(self.color_image, 'person', (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 4, cv2.LINE_AA)
            else:
              cv2.rectangle(self.color_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 4)
              cv2.putText(self.color_image, 'car', (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 4, cv2.LINE_AA) 
        self.position_pub.publish(self.boundingBoxes)
        self.publish_image(self.color_image, height, width)

        if self.visualize :
            cv2.namedWindow('detect_result', cv2.WINDOW_NORMAL)
            # cv2.setWindowProperty('detect_result', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('detect_result', self.color_image)


    def publish_image(self, imgdata, height, width):
        image_temp = Image()
        header = Header(stamp=rospy.Time.now())
        header.frame_id = self.camera_frame
        image_temp.height = height
        image_temp.width = width
        image_temp.encoding = 'bgr8'
        image_temp.data = np.array(imgdata).tobytes()
        image_temp.header = header
        image_temp.step = width * 3
        self.image_pub.publish(image_temp)


def main():
    rospy.init_node('yolov8_ros', anonymous=True)
    yolo_dect = Yolo_Dect()
    rospy.spin()


if __name__ == "__main__":

    main()
