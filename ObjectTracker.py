import cv2
import numpy

class objectSelection(object):
    def __init__(self):
        self.start_coordinates = [0,0]
        self.width = 0
        self.height = 0
        

    def updateSelectionRegion(self,image):
        #image = cv2.imread(image)
        region = cv2.selectROI("Select the area in the frame: ", image)
        self.start_coordinates = [region[0],region[1]]
        self.width = region[2]
        self.height = region[3]
        return region



class videoPlayer(object):
    def __init__(self,video):
        self.capture = cv2.VideoCapture(video)
        if not self.capture.isOpened():
            print("Error opening video file")
        self.frame_width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frame_height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)

    def frameOfObject(self):
        amount_of_frames = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0) # we can switch the 0 to the frame of choice of user
        res, frame = self.capture.read()
        return frame





class trackerObject():
    def __init__(self,frame):
        self.tracker = cv2.legacy.TrackerCSRT_create()
        self.object = objectSelection()
        region = self.object.updateSelectionRegion(frame)
        self.tracker.init(frame,region)

    

   



def main():
    new_video_player = videoPlayer("sample_video.mp4")
    frame = new_video_player.frameOfObject()
    new_tracker = trackerObject(frame)
    while True:
        cap = new_video_player.capture
        ret,frames = cap.read()
        if not ret:
            break
        (success,box)= new_tracker.tracker.update(frames)
        if success:
            (x,y,w,h) = [int(a) for a in box]
            if y <= 0:
                h+=y
                y=0
            if x<=0:
                w+=x
                x=0
            cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)
            roi = frames[y:y+h,x:x+w]
            blur = cv2.GaussianBlur(roi,(51,51),0)
            frames[y:y+h,x:x+w] = blur     
        cv2.imshow("frame", frames)
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


#running the program 
main()

        

        

