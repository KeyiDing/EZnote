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
    
    x_max = int(max(cor[:,:,0])[0])
    x_min = int(min(cor[:,:,0])[0])
    y_max = int(max(cor[:,:,1])[0])
    y_min = int(min(cor[:,:,1])[0])
    
    final = frame[y_min:y_max,x_min:x_max]
    
    
    count = 0
    while(True):
        count += 1
        if count % 100 != 0:
            cap.read()
            continue
        #Capture frame-by-frame
        ret, frame = cap.read()
        
        # add blackboad detection
        img_boxes = detect.detect(frame)
        if len(img_boxes) != 0:
            cv2.drawContours(img_boxes, np.int32([cor]), -1, (0, 255, 0), 2)
        
            square_img = img_boxes[y_min:y_max,x_min:x_max]
            # cv2.imshow('Result',square_img)
            
            if len(final) == 0:
                final = square_img
            else:
                square_img[np.all(square_img == (0, 0, 0), axis=-1)] = final[np.all(square_img == (0, 0, 0), axis=-1)]
                final = square_img
                cv2.imshow('Result',final)
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()