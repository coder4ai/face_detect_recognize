import cv2 as cv
import numpy as np
import os
import sys

def dataset(path):
    label = 0
    imgs, labels = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                filepath = os.path.join(subject_path, filename)
                im = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
                im = cv.resize(im, (200, 200))
                imgs.append(np.asarray(im, dtype=np.uint8))
                labels.append(label)
            label += 1
    return [imgs, labels]

def train(imgs, labels):
    labels = np.asarray(labels, dtype=np.int32)
    model = cv.face.EigenFaceRecognizer_create()
    model.train(np.asarray(imgs), np.asarray(labels))
    return model

def test(model, testimg_path):
    names = ['joe', 'jane', 'jack', 'Mike']# 每个类别实际对应的名称，按类别文件夹顺序
    img = cv.imread(testimg_path)
    #人脸检测
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, 1.1, 1)
    #对图片中每个人脸进行识别
    for (x, y, w, h) in faces:
        img = cv.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        roi = gray[x:x+w, y:y+h]
        roi = cv.resize(roi, (200, 200), interpolation=cv.INTER_LINEAR)
        params = model.predict(roi)
        print("label: %s, confidence: %.3f" % (params[0], params[1]))
        cv.putText(img, names[params[0]], (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv.imshow("img", img)
    cv.waitKey(10000)
    cv.destroyAllWindows()

if __name__ == '__main__':
    data_path = "数据集文件夹根路径"
    testimg_path = "测试图片路径"
    imgs, labels = dataset(data_path)
    model = train(imgs, labels)
    test(model, testimg_path)
