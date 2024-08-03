import numpy as np
import cv2
import face_recognition
import os

class ImageWithID:
    def __init__(self, image, image_id: int):
        self.image_id = image_id
        self.faces_IDs = []
        self.image = image
   
class IDwithencodings:
    def __init__(self,image_id : int, encodings):
        self.image_id = image_id
        self.encodings = encodings
    
class Person:
    def __init__(self,face_id : int, encodings):
        self.face_id = face_id
        self.encodings = encodings
        
def Creat_new_face(faces_id_count,encoding,Known_Faces,ImageWithID):
    new_Person = Person(faces_id_count,encoding)
    Known_Faces.append(new_Person)
    ImageWithID.faces_IDs.append(faces_id_count)
    return (faces_id_count + 1)

path = 'Images database'

image_id_count = 1
faces_id_count = 1

image_dict = {}
encodings_of_images = []
Known_Faces = []

mylist = os.listdir(path)

for n in mylist:
    current_im = cv2.imread(f"{path}/{n}")
    current_img_class = ImageWithID(current_im, image_id_count)
    image_dict[image_id_count] = current_img_class
    image_id_count += 1

for n in image_dict.values():
    encoding = n.image
    encoding = cv2.cvtColor(encoding,cv2.COLOR_RGB2BGR)
    encoding = face_recognition.face_encodings(encoding)
    if encoding:   
        encodings_of_images.append(IDwithencodings(n.image_id, encoding))

for n in encodings_of_images:
    j = n.encodings
    for l in j:
        result = face_recognition.compare_faces([obj.encodings for obj in Known_Faces],l)
        if result == [] or np.all(np.logical_not(result)):
            faces_id_count = Creat_new_face(faces_id_count,l,Known_Faces,image_dict[n.image_id])
        else:
            for index, k in enumerate(result):
                if k == True:
                    image_dict[n.image_id].faces_IDs.append(Known_Faces[index].face_id)


print(image_dict[1].faces_IDs)
print(image_dict[2].faces_IDs)
print(image_dict[3].faces_IDs)
print(image_dict[4].faces_IDs)
