import cv2 as cv
import numpy as np

def face_detect(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_detector.detectMultiScale(gray, 1.02, 20)
    for x, y, w, h in faces:
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv.imshow("face_detect", image)

def video_face_detect():
    capture = cv.VideoCapture(0)
    while True:
        ret, frame = capture.read()
        frame = cv.flip(frame, 1)
        face_detect(frame)
        c = cv.waitKey(10)
        if c==27:  # ESC
            break

if __name__ == '__main__':
    src = cv.imread("test.jpg")
    face_detect(src)
    # video_face_detect()
    cv.waitKey(10000)
    cv.destroyAllWindows()
