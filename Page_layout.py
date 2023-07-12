
import streamlit as st
from PIL import Image
from sklearn import datasets
import pandas as pd


class main_page(object):
    def __init__(self) :
        #####Main Page s
        self.image=Image.open('media/bedo2.png')
        st.image(self.image,width=250)
        st.title('Bedo AI Computer Vision Virtual labs')
        #sidebar Content
        st.sidebar.image('media/nural2.png',width=50)
        st.sidebar.header("What do you want to learn today")
        self.categories=st.sidebar.selectbox("Computer vision Virtuqal labs ",options=["Computer Vision"])
        #self.Algotypes=st.sidebar.selectbox("Agorithm Type",options=['Regression','Classification'])
        self.learntype=st.sidebar.radio("learning enviornment",options=['Tutorials','Algorithm in Action','Applied projects'])
        new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 24px;">'+" 👉" + self.categories +  '</p>'
        st.markdown(new_title, unsafe_allow_html=True)
    def select_dataset():
        Datasets=st.sidebar.selectbox("Datasets",options=["Select Dataset ",'Image classifier',
                                                            'Digit recognition','Face Recognition','Face Attendance',"Traffic sign identification",'Object Detection','Image Effects',
                                                            'Video Effects',"Object Tracking",
                                                            "Text Detection","Face Mask Detect","QR Code Reader"])
        return Datasets
    def alignh(lines,colname):
       for i in range (1 , lines):
         colname.markdown("#")
