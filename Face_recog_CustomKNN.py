import cv2
import numpy as np
import os
import math

#KNN Algorithm
def dist(x1, x2):
    distance = 0
    for i in range(len(x1)):
        distance += (x1[i] - x2[i]) ** 2
    return math.sqrt(distance)
    
def KNN(train, test, k=2):
    X_train = [row[:-1] for row in train]
    Y_train = [row[-1] for row in train]
    vals = []

    for i in range(len(X_train)):
        vals.append((dist(test, X_train[i]), Y_train[i]))

    # sort the vals on the basis of distance , take only first 'k' values as they will be the nearest ones
    vals = sorted(vals)[:k]

    #finding which class id is more closer
    class_counts = {}
    for val in vals:
        if val[1] in class_counts:
            class_counts[val[1]] += 1
        else:
            class_counts[val[1]] = 1

    # Finding the class with the highest count
    max_count = -1
    predicted_class = None
    for cls, count in class_counts.items():
        if count > max_count:
            max_count = count
            predicted_class = cls

    return(int(predicted_class))


face_data=[]

labels=[]

class_id=0

names={}

for file in os.listdir():
    if file.endswith('.npy'):
        data_item=np.load(file)
        face_data.append(data_item)
        
        target=np.ones((data_item.shape[0],))*class_id
        labels.append(target)
        
        names[class_id]=file[:-4]
        class_id+=1

face_data=np.concatenate(face_data,axis=0)
face_label=np.concatenate(labels,axis=0).reshape((-1,1))

training_data=np.append(face_data,face_label,axis=1)

cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
    boolean,frame=cap.read()
    if boolean==False:
        continue
    
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    
    for (x,y,w,h) in faces:
        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))

        face_section=face_section.flatten()
        
        output=KNN(training_data,face_section,k=5)
        predicted_name=names[output]
        
        cv2.putText(frame,predicted_name,(x+10,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    
    cv2.imshow("Prediction ",frame)

    keypressed=cv2.waitKey(1) & 0xFF
    if keypressed==ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
