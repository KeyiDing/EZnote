import tensorflow_hub as hub
import cv2
import numpy
import tensorflow as tf
import pandas as pd
import numpy as np
import imutils

# Carregar modelos
detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
labels = pd.read_csv('~/EZnote/human_detection/labels.csv',sep=';',index_col='ID')
labels = labels['OBJECT (2017 REL.)']

# video IO

def detect(frame):
    #Resize to respect the input_shape
    # inp = imutils.resize(frame, height=512)
    inp = frame

    
    # cv2.imwrite("frame.png", frame)

    #Convert img to RGB
    rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

    #Is optional but i recommend (float convertion and convert img to tensor image)
    rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)

    #Add dims to rgb_tensor
    rgb_tensor = tf.expand_dims(rgb_tensor , 0)
    
    boxes, scores, classes, num_detections = detector(rgb_tensor)
    
    pred_labels = classes.numpy().astype('int')[0]
    
    pred_labels = [labels[i] for i in pred_labels]
    pred_boxes = boxes.numpy()[0].astype('int')
    pred_scores = scores.numpy()[0]
    #loop throughout the faces detected and place a box around it
    
    img_boxes = []
    
    for score, (ymin,xmin,ymax,xmax), label in zip(pred_scores, pred_boxes, pred_labels):
        if score < 0.5 or label != "person":
            continue          
    
        xmax += 0.05*(xmax-xmin)
        xmax = int(xmax)
        if xmax > frame.shape[1]-1:
            xmax = frame.shape[1]-1
        xmin -= 0.05*(xmax-xmin)
        xmin = int(xmin)
        if xmin < 0:
            xmin = 0
            
        ymax += 0.05*(ymax-ymin)
        ymax = int(ymax)
        if ymax > frame.shape[0]-1:
            ymax = frame.shape[0]-1
        ymin -= 0.05*(ymax-ymin)
        ymin = int(ymin)
        if ymin < 0:
            ymin = 0
            
        points = np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])
        img_boxes = cv2.fillPoly(frame, pts = [points], color =(0,0,0))
  
    return img_boxes