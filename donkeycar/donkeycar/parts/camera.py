import os
import time
import numpy as np
from PIL import Image
import glob
from donkeycar.utils import rgb2gray
import cv2

class BaseCamera:

    def run_threaded(self):
        return self.frame

class MutilCamera(BaseCamera):
    def __init__(self, image_w=160, image_h=120, image_d=3, framerate=20, iCam=0):
        import pygame
        import pygame.camera

        from picamera.array import PiRGBArray
        from picamera import PiCamera

        import VL53L1X2 as vl
        import RPi.GPIO as GPIO

        import time

        super().__init__()


        self.image_d = image_d
        self.resolution = (image_w, image_h)
        self.image_w, self.image_h = image_w, image_h

        pygame.init()
        pygame.camera.init()
        l = pygame.camera.list_cameras()
        print('cameras', l)
        self.cam = pygame.camera.Camera(l[iCam], self.resolution, "RGB")
        self.cam.start()
        self.framerate = framerate

        # ==================================
        # initialize the camera and stream
        self.camera = PiCamera()  # PiCamera gets resolution (height, width)
        self.camera.resolution = self.resolution
        self.camera.framerate = framerate
        self.rawCapture = PiRGBArray(self.camera, size=self.resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture, format="rgb", use_video_port=True)


        #--------------
        SHUTX_PIN_1 = 20
        SHUTX_PIN_2 = 16

        GPIO.setwarnings(False)

        # Setup GPIO for shutdown pins on each VL53L0X
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(SHUTX_PIN_1, GPIO.OUT)
        GPIO.setup(SHUTX_PIN_2, GPIO.OUT)

        # Set all shutdown pins low to turn off each VL53L0X
        GPIO.output(SHUTX_PIN_1, GPIO.LOW)
        GPIO.output(SHUTX_PIN_2, GPIO.LOW)


        # Start with first sensor
        pin = SHUTX_PIN_1
        GPIO.output(pin, GPIO.HIGH)

        self.tof = vl.VL53L1X(i2c_bus=1)
        self.tof.open()
        self.tof.add_sensor(sensor_id=1, address=0x30)
        self.tof.change_address(1, 0x29)
        GPIO.output(pin, GPIO.LOW)
        pin = SHUTX_PIN_2
        GPIO.output(pin, GPIO.HIGH)
        self.tof.add_sensor(sensor_id=2, address=0x30)
        GPIO.output(SHUTX_PIN_1, GPIO.HIGH)

        self.tof.start_ranging(1)
        self.tof.start_ranging(2)

        #--------------------------

        # ==================================
        # initialize variable used to indicate
        # if the thread should be stopped

        self.frame = np.zeros((image_h, image_w, image_d)).astype(int) * 255
        self.fusion = np.zeros((image_h, image_w, image_d)).astype(int) * 255
        self.on = True
        print('WebcamVideoStream and PiCamera loaded.. .warming camera')

        time.sleep(2)

    def update(self):
        from datetime import datetime, timedelta
        import pygame.image
        while self.on:
            start = datetime.now()

            # WebCamera: frame1
            if self.cam.query_image():
                snapshot = self.cam.get_image()
                snapshot1 = pygame.transform.scale(snapshot, self.resolution)
                frame1 = pygame.surfarray.pixels3d(
                    pygame.transform.rotate(pygame.transform.flip(snapshot1, True, False), -90))

                # Import opencv for mask generation
                greenLower = (30, 45, 70)  # Set boundary for green (HSV Color Space)
                greenUpper = (64, 255, 255)
                img = frame1.copy()  # Import image
                blurred = cv2.GaussianBlur(img, (11, 11), 0)
                hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)  # Transfer to HSV color space
                mask = cv2.inRange(hsv, greenLower, greenUpper)
                mask = cv2.erode(mask, None, iterations=2)  # Create mask to find green areas
                mask = cv2.dilate(mask, None, iterations=2)
                cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts[1] == []:
                    x, y, radius = [0, 0, 0]
                else:
                    c = max(cnts[1], key=cv2.contourArea)
                    ((y, x), radius) = cv2.minEnclosingCircle(c)  # get center and radius of the ball
                    # img = cv2.circle(img, (int(y), int(x)), int(radius), (255, 0, 0), 2)
                    for row in range(mask.shape[0]):
                        for col in range(mask.shape[1]):
                            if (row-x)**2 + (col-y)**2 > radius**2:
                                mask[row][col] = 0
                            else:
                                mask[row][col] = 255
        

                frame1 = rgb2gray(frame1)

            # PiCamera: frame2
            f = next(self.stream)
            frame2 = rgb2gray(f.array)
            self.rawCapture.truncate(0)

            # Sensor info
            Dist_M = np.zeros((self.image_h, self.image_w))
            distance_in_mm1 = self.tof.get_distance(1)
            distance_in_mm2 = self.tof.get_distance(2)
            d1 = np.round(min(distance_in_mm1 / 1000 * 255, 255))
            d2 = np.round(min(distance_in_mm2 / 1000 * 255, 255))
            Dist_M[:, :int(self.image_w / 2)] = int(d1)
            Dist_M[:, int(self.image_w / 2):] = int(d2)


            # Fusion of both cameras, testing
            # ??============================
            self.fusion[:, :, 0] = frame2
            self.fusion[:, :, 1] = mask
            self.fusion[:, :, 2] = Dist_M


            self.frame = self.fusion

            # print("1: {}, 2: {}, reso: {}".format(frame1.dtype, frame2.dtype, self.resolution))

            # =============================
            stop = datetime.now()
            s = 1 / self.framerate - (stop - start).total_seconds()
            if s > 0:
                time.sleep(s)

        self.cam.stop()



    def run(self):
        f = next(self.stream)
        frame = f.array
        self.rawCapture.truncate(0)
        if self.image_d == 1:
            frame = rgb2gray(frame)
        return frame

    def shutdown(self):
        # indicate that the thread should be stopped
        self.on = False
        print('stoping Webcam and PiCamera')
        time.sleep(.5)
        self.stream.close()
        self.rawCapture.close()
        self.camera.close()


class PiCamera(BaseCamera):
    def __init__(self, image_w=160, image_h=120, image_d=3, framerate=20):
        from picamera.array import PiRGBArray
        from picamera import PiCamera

        resolution = (image_w, image_h)
        # initialize the camera and stream
        self.camera = PiCamera()  # PiCamera gets resolution (height, width)
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,
                                                     format="rgb", use_video_port=True)

        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.on = True
        self.image_d = image_d

        print('PiCamera loaded.. .warming camera')
        time.sleep(2)

    def run(self):
        f = next(self.stream)
        frame = f.array
        self.rawCapture.truncate(0)
        if self.image_d == 1:
            frame = rgb2gray(frame)
        return frame

    def update(self):
        # keep looping infinitely until the thread is stopped
        for f in self.stream:
            # grab the frame from the stream and clear the stream in
            # preparation for the next frame
            self.frame = f.array
            self.rawCapture.truncate(0)

            if self.image_d == 1:
                self.frame = rgb2gray(self.frame)

            # if the thread indicator variable is set, stop the thread
            if not self.on:
                break

    def shutdown(self):
        # indicate that the thread should be stopped
        self.on = False
        print('stoping PiCamera')
        time.sleep(.5)
        self.stream.close()
        self.rawCapture.close()
        self.camera.close()


class Webcam(BaseCamera):
    def __init__(self, image_w=160, image_h=120, image_d=3, framerate = 20, iCam = 0):
        import pygame
        import pygame.camera

        super().__init__()
        resolution = (image_w, image_h)
        pygame.init()
        pygame.camera.init()
        l = pygame.camera.list_cameras()
        print('cameras', l)
        self.cam = pygame.camera.Camera(l[iCam], resolution, "RGB")
        self.resolution = resolution
        self.cam.start()
        self.framerate = framerate

        # initialize variable used to indicate
        # if the thread should be stopped
        self.frame = np.ones((image_h, image_w, image_d)).astype(int)*255
        self.on = True
        self.image_d = image_d

        print('WebcamVideoStream loaded.. .warming camera')

        time.sleep(2)

    def update(self):
        from datetime import datetime, timedelta
        import pygame.image 

        while self.on:
            start = datetime.now()

            if self.cam.query_image():
                # snapshot = self.cam.get_image()
                # self.frame = list(pygame.image.tostring(snapshot, "RGB", False))
                snapshot = self.cam.get_image()
                snapshot1 = pygame.transform.scale(snapshot, self.resolution)
                #************************
                snapshot2 = pygame.transform.rotate(pygame.transform.flip(snapshot1, True, False), -90)
                frame1 = pygame.surfarray.pixels3d(snapshot2)
                greenLower = (30, 45, 70)  # Set boundary for green (HSV Color Space)
                greenUpper = (64, 240, 240)
                img = frame1.copy()  # Import image
                blurred = cv2.GaussianBlur(img, (11, 11), 0)
                hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)  # Transfer to HSV color space
                mask = cv2.inRange(hsv, greenLower, greenUpper)
                mask = cv2.erode(mask, None, iterations=2)  # Create mask to find green areas
                mask = cv2.dilate(mask, None, iterations=2)
#                cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                if cnts[1] == []:
#                    x, y, radius = [0, 0, 0]
#                else:
#                    c = max(cnts[1], key=cv2.contourArea)
#                    ((y, x), radius) = cv2.minEnclosingCircle(c)  # get center and radius of the ball
#                    # img = cv2.circle(img, (int(y), int(x)), int(radius), (255, 0, 0), 2)
#                    for row in range(mask.shape[0]):
#                        for col in range(mask.shape[1]):
#                            if (row-x)**2 + (col-y)**2 > radius**2:
#                                mask[row][col] = 0
#                            else:
#                                mask[row][col] = 255
                img[:,:,0] = mask
                self.frame = img
#                for i in range(3):
#                    self.frame[:,:,i] = mask
           #*********
                if self.image_d == 1:
                    self.frame = rgb2gray(self.frame)

            stop = datetime.now()
            s = 1 / self.framerate - (stop - start).total_seconds()
            if s > 0:
                time.sleep(s)

        self.cam.stop()

    def run_threaded(self):

        return self.frame

    def shutdown(self):
        # indicate that the thread should be stopped
        self.on = False
        print('stoping Webcam')
        time.sleep(.5)


class MockCamera(BaseCamera):
    '''
    Fake camera. Returns only a single static frame
    '''
    def __init__(self, image_w=160, image_h=120, image_d=3, image=None):
        if image is not None:
            self.frame = image
        else:
            self.frame = Image.new('RGB', (image_w, image_h))

    def update(self):
        pass

    def shutdown(self):
        pass


class ImageListCamera(BaseCamera):
    '''
    Use the images from a tub as a fake camera output
    '''
    def __init__(self, path_mask='~/d2/data/**/*.jpg'):
        self.image_filenames = glob.glob(os.path.expanduser(path_mask), recursive=True)
    
        def get_image_index(fnm):
            sl = os.path.basename(fnm).split('_')
            return int(sl[0])

        '''
        I feel like sorting by modified time is almost always
        what you want. but if you tared and moved your data around,
        sometimes it doesn't preserve a nice modified time.
        so, sorting by image index works better, but only with one path.
        '''
        self.image_filenames.sort(key=get_image_index)
        #self.image_filenames.sort(key=os.path.getmtime)
        self.num_images = len(self.image_filenames)
        print('%d images loaded.' % self.num_images)
        print( self.image_filenames[:10])
        self.i_frame = 0
        self.frame = None
        self.update()

    def update(self):
        pass

    def run_threaded(self):        
        if self.num_images > 0:
            self.i_frame = (self.i_frame + 1) % self.num_images
            self.frame = Image.open(self.image_filenames[self.i_frame]) 

        return np.asarray(self.frame)

    def shutdown(self):
        pass

# 18 45 164
# 64 255 255