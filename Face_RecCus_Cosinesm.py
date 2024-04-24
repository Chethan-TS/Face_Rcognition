import cv2
import numpy as np
import os
import math
import heapq

#KNN with distannce calculated using cosine similarity instead of Euclidean
# Define dot product function
def dot_product(x1, x2):
    return sum(x * y for x, y in zip(x1, x2))

# Define magnitude function
def magnitude(x):
    return math.sqrt(sum(val ** 2 for val in x))

# Define cosine similarity function
def cosine_similarity(x1, x2):
    return dot_product(x1, x2) / (magnitude(x1) * magnitude(x2) + 1e-9)

# Define KNN algorithm using cosine similarity
def KNN(train, test, k=5):
    X_train = [row[:-1] for row in train]
    Y_train = [row[-1] for row in train]
    magnitudes_train = [magnitude(x) for x in X_train]
    heap = []

    for i in range(len(X_train)):
        cosine_sim = cosine_similarity(test, X_train[i])
        if cosine_sim > 0 and magnitudes_train[i] > 0:
            heapq.heappush(heap, (cosine_sim, Y_train[i]))
            if len(heap) > k:
                heapq.heappop(heap)

    class_counts = {}
    for _, cls in heap:
        class_counts[cls] = class_counts.get(cls, 0) + 1

    # Find the class with the highest count
    predicted_class = max(class_counts, key=class_counts.get)

    return predicted_class

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
