#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src import utils
import time
import cv2

import matplotlib.pyplot as plt
from colour import Color

from os.path import basename 
import sys

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False #设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法,经验证影响不大
vis = False
save_output = True

data_path =  './video/'
model_path = './final_models/cmtl_qn_part2_1200.h5'

net = CrowdCounter()

trained_model = os.path.join(model_path)
network.load_net(trained_model, net)
net.cuda()
net.eval()

fff = time.time()
count = 0

density_range = 1000
gradient = np.linspace(0, 1, density_range)
cmap = plt.get_cmap("rainbow") 
initial = Color("blue") 
hex_colors = list(initial.range_to(Color("red"), density_range))
rgb_colors = [[rgb * 255 for rgb in color.rgb] for color in hex_colors]

es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 3))

starttime = time.time()
for image in os.listdir(data_path):
    video_path = os.path.join(data_path,image)
    print(video_path)
    cap = cv2.VideoCapture(video_path)
    print("OK1!")
    fps = cap.get(cv2.CAP_PROP_FPS)  
    #fps =1
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  
    fourcc =  cv2.VideoWriter_fourcc(*'X264')
    print(basename(video_path))
    vwrite = cv2.VideoWriter('lc-'+basename(video_path) + '_2.mp4', fourcc, fps, size)
    success, img_color = cap.read()
    print("OK2!")
    while (success):
        print("OK3!")
        count += 1
        #if(count%5!=0):
        #    success, img_color = cap.read()
        #    continue

        #step 1 获取密度图+人数
        img = cv2.resize(img_color,(0,0), fx=0.4, fy=0.4)
        #img = cv2.resize(img_color,(0,0), fx=1.0, fy=1.0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # gray
        img = img.astype(np.float32, copy=False)
        ht = img.shape[0]
        wd = img.shape[1]
        ht_1 = int(ht/4)*4
        wd_1 = int(wd/4)*4
        if(ht!=ht_1 or wd!=wd_1):
            img = cv2.resize(img,(wd_1,ht_1))
        im_blob = img.reshape((1,1,img.shape[0],img.shape[1]))
        density_map = net(im_blob, [])
        density_map = density_map.data.cpu().numpy()
        et_count = np.sum(density_map)
        print(et_count)
 
        #step 2 绘制热力图
        dm = 255*density_map[0][0]*1000
        rows, cols = dm.shape
        color_map = np.empty([rows, cols, 3], dtype=np.uint8)
        for i in range(rows):
            for j in range(cols):
                ratio = min(density_range - 1, int(dm[i][j]))
                for k in range(3):
                    color_map[i][j][k] = int(rgb_colors[ratio][k])
        color_map = cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR)
        color_map = cv2.blur(color_map,(3,3))
        heatmap = cv2.resize(color_map, (img_color.shape[1], img_color.shape[0]))
        alpha = 0.3
        cv2.addWeighted(heatmap, alpha, img_color, 1-alpha, 0, img_color)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_color, 'num : '+str(int(et_count)),(50,50), font, 1, (0,0,255), 4, cv2.LINE_AA)

        #step 3 当前帧报警
        #当外接矩形区域超过一定阈值则警示，区域连续15帧警示，则输出报警状态，且将警示区域的外接轮廓在框出来
        foreground =255*density_map/np.max(density_map)
        foreground= foreground[0][0]
        foreground = cv2.threshold(foreground, 30, 255, cv2.THRESH_BINARY)[1]
        foreground = cv2.dilate(foreground, es, iterations=1)
        foreground = cv2.erode(foreground,es,iterations = 1)
        foreground = cv2.dilate(foreground, es, iterations=5)

        foreground_map = np.empty([rows, cols, 1], dtype=np.uint8)
        for i in range(rows):
            for j in range(cols):
                foreground_map[i][j] = int(foreground[i][j])

        foreground_resize = cv2.resize(foreground_map, (img_color.shape[1], img_color.shape[0]))

        image, contours, hierarchy = cv2.findContours(foreground_resize.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cr in contours:
            if cv2.contourArea(cr) < 4000:
                continue
            (x, y, w, h) = cv2.boundingRect(cr)
            cv2.rectangle(foreground_resize, (x, y), (x+w, y+h), (128, 128, 128), 2)
            #print(w/foreground_resize.shape[1],h/foreground_resize.shape[0])
            if(w>0.65*foreground_resize.shape[1] or h>0.65*foreground_resize.shape[0] or (w>0.3*foreground_resize.shape[1] and h>0.25*foreground_resize.shape[0])):
                cv2.rectangle(foreground_resize, (x, y), (x+w, y+h), (255, 255, 255), 2)
                print("#############OK")
                #cv2.rectangle(img_color, (x, y), (x+w, y+h), (0, 0, 255), 3)
                cr_l = []
                cr_l.append(cr)
                cv2.drawContours(img_color, cr_l, -1, (0,0,255), thickness=3)
                cv2.imwrite("./output2/%d.jpg" %(count),img_color)
        #cv2.imwrite("./output2/__%d.jpg" %(count),foreground)

        #step4 保存
        cv2.imwrite("./output2/%d.jpg" %(count),img_color)
        vwrite.write(img_color)
        success, img_color = cap.read()
        #if(count>=100):
        if(count>=30000):
            break
    cap.release()
    vwrite.release()

endtime = time.time()
#print( 'process 1 speed: {:.3f}s / iter'.format(float(endtime - starttime)/count))