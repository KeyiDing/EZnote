from genericpath import exists
from site import venv
import numpy as np
import cv2
import numpy
import sys
import os

from image_process.formatImg import ImgtoNote
sys.path.append("./human_detection/")
import detect

sys.path.append("./image_process/")
import formatImg

def note(video_path):
    video_list = []
    # cap = cv2.VideoCapture("Movie on 2022-9-17 at 2.19 PM.mov")
    cap = cv2.VideoCapture(video_path)
    
    ret, frame = cap.read()
    cor = formatImg.findcoordinates(frame)
    while (len(cor)==0):
        ret, frame = cap.read()
        cor = formatImg.findcoordinates(frame)
    
    x_max = int(max(cor[:,:,0])[0])
    x_min = int(min(cor[:,:,0])[0])
    y_max = int(max(cor[:,:,1])[0])
    y_min = int(min(cor[:,:,1])[0])
    
    final = frame[y_min:y_max,x_min:x_max]
    
    
    count = 0
    while(True):
        count += 1
        if count % 1 != 0:
            cap.read()
            continue
        #Capture frame-by-frame
        ret, frame = cap.read()
        if frame is None:
            break
        
        # add blackboad detection
        img_boxes,box = detect.detect(frame)
        if len(img_boxes) != 0:
            cv2.drawContours(img_boxes, np.int32([cor]), -1, (0, 255, 0), 2)
            # cv2.imshow('Result',img_boxes)
        
            square_img = img_boxes[y_min:y_max,x_min:x_max]
            # cv2.imshow('Result',square_img)
            
            if len(final) == 0:
                final = square_img
            else:
                square_img[np.all(square_img == (0, 0, 0), axis=-1)] = final[np.all(square_img == (0, 0, 0), axis=-1)]
                box[y_min:y_max,x_min:x_max] = square_img
                final = square_img
                video_list.append(box)
                cv2.imshow('Result',box)
                
        # if count % 100 == 0:  
        #     if count / 100 == 1:  
        #         new_cor = np.copy(cor)
        #         new_cor = new_cor.reshape((4,2))
        #         new_cor[:,0] -= x_min
        #         new_cor[:,1] -= y_min
        #     note_img = ImgtoNote(final, new_cor)
        #     cv2.imshow("Result",note_img)
        #     cv2.imwrite("./notes/note_{}.png".format(int(count/300)),note_img)
        #     cv2.imwrite("final.png",final)
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    height,width,layers=video_list[1].shape
    video_list = np.array(video_list)
    video=cv2.VideoWriter('video_1.mp4',cv2.VideoWriter_fourcc('M','J','P','G'),10,(width,height))
    
    for i in range(len(video_list)):
        video.write(video_list[i])
    video.release()
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    note("video1630936765.mp4")