import cv2
import numpy as np
import os
import math

#-------------------------------KNN STARTS-----------------------------------------

# Define distance function
def dist(x1, x2):
    distance = 0
    for i in range(len(x1)):
        distance += (x1[i] - x2[i]) ** 2
    return math.sqrt(distance)

# Define KNN algorithm
def KNN(train, test, k=2):
    # X will have all the features except the last one as the last one will contain the ID
    X_train = [row[:-1] for row in train]
    Y_train = [row[-1] for row in train]
    # Creating a list to save the value of distance between test and training point and will save id also
    vals = []

    for i in range(len(X_train)):
        vals.append((dist(test, X_train[i]), Y_train[i]))

    # Now we will sort the vals on the basis of distance and will take only first 'k' values from it as they
    # will be the nearest ones
    vals = sorted(vals)[:k]

    # Now we will find which class id is more closer to the test image.
    class_counts = {}
    for val in vals:
        if val[1] in class_counts:
            class_counts[val[1]] += 1
        else:
            class_counts[val[1]] = 1

    # Find the class with the highest count
    max_count = -1
    predicted_class = None
    for cls, count in class_counts.items():
        if count > max_count:
            max_count = count
            predicted_class = cls

    return(int(predicted_class))
    
#--------------------------------KNN ENDS----------------------------------------------------------

face_data=[]

#This will an array which will save class ids.
labels=[]

#defining class id which will allocated to every image.
class_id=0

#A dcitionary is defined here which will treat class id as key and its value will be the name of the file
names={}

for file in os.listdir():
    if file.endswith('.npy'):
        #Loading the training data one at a time
        data_item=np.load(file)
        face_data.append(data_item)
        
        target=np.ones((data_item.shape[0],))*class_id
        labels.append(target)
        
        names[class_id]=file[:-4]
        #incrementing class id after each iteration
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
        
        #Now calling the KNN to predict the output
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
