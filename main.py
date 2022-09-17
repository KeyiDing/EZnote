import cv2
import sys
sys.path.append("./human_detection/")
import detect

def main():
    cap = cv2.VideoCapture("Movie on 2022-9-17 at 1.53 PM #2.mov")
    
    while(True):
        #Capture frame-by-frame
        ret, frame = cap.read()
        
        # add blackboad detection
        
        frame,img_boxes = detect.detect(frame)
        if len(img_boxes) != 0:
            cv2.imshow('Result',img_boxes)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()