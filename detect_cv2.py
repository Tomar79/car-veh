import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import os
import shutil

st.set_page_config(layout='wide')
st.image('banner.jpg')
st.title("Human - Vehcles analyzer")
frozen_model = 'frozen_inference_graph.pb'
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
model = cv2.dnn_DetectionModel(frozen_model, config_file)
model.setInputSize(320, 320)
model.setInputScale(1.0/127.5) # 255 / 2 = 127.5
model.setInputMean((127.5, 127.5, 127.5)) # mobilenet => [-1, 1]
model.setInputSwapRB(True)
file_name = 'COCO_labels.txt'
with open(file_name, 'rt') as fpt:
    classlabels = fpt.read().rstrip('\n').split('\n')
classes = [ 'person', 'bicycle', 'car', 'motorcycle','bus', 'train', 'truck']



def inspect(img,file_name,saving_dir='',source_dir=''): 
    img = np.array(img)
    ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)
    font_scale = 2
    font = cv2.FONT_HERSHEY_PLAIN
    detect_human=[]
    detect_vehcl=[]
    for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
        if classlabels[ClassInd-1] in classes:
            cv2.rectangle(img, boxes, (255, 255, 0), 2)
            cv2.putText(img, classlabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale=font_scale, color=(0, 255, 255), thickness=2)
            if classlabels[ClassInd-1]=='person':
                detect_human.append(classlabels[ClassInd-1])
            else:
                detect_vehcl.append(classlabels[ClassInd-1])
    if len(detect_human)>0 or len(detect_vehcl)>0:
        file_path=saving_dir+'/'+file_name
        cv2.imwrite(file_path, img)
    else:
        file_path=source_dir+'/'+file_name
        os.remove(file_path)
    return img



col1,col2=st.columns((5,5))
# if 'dirname' not in st.session_state:
#         st.session_state['dirname'] = ''
col1,col2=st.columns((5,5))
# Set source Image directory and  Annotate Images for dataset
select_dir=col1.button("Select Images source location ")
if select_dir :
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    sdirname =  filedialog.askdirectory(master=root)  
    sdirname.replace ('\\',"/")     
    st.session_state['sdirname'] = sdirname
    col1.subheader(f'Source Images directory {sdirname}')
    savlocprint=col2.empty()
    ddirname='c:/analyzed'
    if os.path.exists(ddirname):
        pass
    else:
        os.makedirs(ddirname)
    files = os.listdir(sdirname)
    # Loop through each file in the directory
    imag_view=col2.empty()
    import time
    for file in files:
        # Check if the file is an image file
        if file.endswith('.jpg') or file.endswith('.png'):
            # Open the image file using PIL
            img_path = os.path.join(sdirname, file)
            img = Image.open(img_path)
            img1=img.resize((480,480),resample= Image.Resampling.NEAREST)
            inspected_img=inspect(img1,file,ddirname,sdirname)
            # Do something with the image (e.g., display it)
            imag_view.image(inspected_img,file)
            time.sleep(0.5)


     

