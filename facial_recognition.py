import numpy as np
import cv2
import face_recognition
import os

"""""
imgelon = face_recognition.load_image_file('Elon_Musk.jpg')
imgelon_BGR = cv2.cvtColor(imgelon,cv2.COLOR_RGB2BGR)

imgelontest = face_recognition.load_image_file('Elon_Musk_test.jpg')
imgelontest_BGR = cv2.cvtColor(imgelontest,cv2.COLOR_RGB2BGR)

encodingelon = face_recognition.face_encodings(imgelon)[0]
encodingtest = face_recognition.face_encodings(imgelontest)[0]

result = face_recognition.compare_faces([encodingelon],encodingtest)
facedistance = face_recognition.face_distance([encodingelon],encodingtest)

cv2.imshow('Elon_Musk',imgelon_BGR)
cv2.imshow('Elon_Musk_test',imgelontest_BGR)
cv2.waitKey(0)

"""""

path = 'Images database'
images = []
encodings_of_images = []
mylist = os.listdir(path)

for n in mylist:
    encoding = face_recognition.load_image_file('Images database/' +n)
    encoding = cv2.cvtColor(encoding,cv2.COLOR_RGB2BGR)
    encoding = face_recognition.face_encodings(encoding)[0]
    encodings_of_images.append(encoding)
    print(n)

for index, n in enumerate(encodings_of_images):
    result = face_recognition.compare_faces(encodings_of_images,n)
    
