import pickle
import cv2
import sys
import time


def main():
    
    if len(sys.argv) != 3:
        print("Invalid number of arguments. Expecting <classification path> <video path>")
        sys.exit(-1)
    class_path = sys.argv[1]
    video_path = sys.argv[2]

    classif = pickle.load(open(class_path, 'rb'))
    cap = cv2.VideoCapture(video_path)
    i = 0
    while(True):
        
        # Capture frames in the video
        ret, frame = cap.read()
    
        # describe the type of font
        # to be used.
        font = cv2.FONT_HERSHEY_SIMPLEX
    
        # Use putText() method for
        # inserting text on video
        cv2.putText(frame, 
                    classif[i], 
                    (50, 50), 
                    font, 1, 
                    (0, 255, 255), 
                    2, 
                    cv2.LINE_4)
        i += 1
        # Display the resulting frame
        cv2.imshow('video', frame)
        time.sleep(0.05)
        # creating 'q' as the quit 
        # button for the video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # release the cap object
    cap.release()
    # close all windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()