import numpy as np
import cv2
import numpy
import sys
sys.path.append("./human_detection/")
import detect

sys.path.append("./image_process/")
import formatImg

def main():
    cap = cv2.VideoCapture("Movie on 2022-9-17 at 2.19 PM.mov")
    
    ret, frame = cap.read()
    cor = formatImg.findcoordinates(frame)
    print(cor)
    
    while(True):
        #Capture frame-by-frame
        ret, frame = cap.read()
        
        # add blackboad detection
        img_boxes = detect.detect(frame)
        if len(img_boxes) != 0:
            cv2.drawContours(img_boxes, np.int32([cor]), -1, (0, 255, 0), 2)
            cv2.imshow('Result',img_boxes)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()