import threading
import cv2

class Webcam(cv2.VideoCapture):
    def __init__(self, camera_index=0):
        super().__init__(camera_index)

        self.ret, self.frame = super().read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self
    
    def update(self):
        while True:
            if self.stopped:
                self.release()
                return
            
            self.ret, self.frame = super().read()
    
    def read(self):
        return self.ret, self.frame
    
    def stop(self):
        self.stopped = True
