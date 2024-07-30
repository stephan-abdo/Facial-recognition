import numpy as np
import cv2
import face_recognition

imgelon = face_recognition.load_image_file('Elon_Musk.jpg')
imgelon_BGR = cv2.cvtColor(imgelon,cv2.COLOR_RGB2BGR)

imgelontest = face_recognition.load_image_file('Elon_Musk_test.jpg')
imgelontest_BGR = cv2.cvtColor(imgelontest,cv2.COLOR_RGB2BGR)

faceloc = face_recognition.face_locations(imgelon)
#encodingelon = face_recognition.face_encodings(imgelon)
#cv2.rectangle(imgelon_BGR,faceloc[1],faceloc[2],faceloc[3],faceloc[0],(255,0,255),2)

cv2.imshow('Elon_Musk',imgelon_BGR)
cv2.imshow('Elon_Musk_test',imgelontest_BGR)
cv2.waitKey(0)
