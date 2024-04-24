import cv2
import numpy as np

#Intializing camera(webcam here)
cap=cv2.VideoCapture(0)

#making a skip variable whose purpose is defined in line number 71.
skip=0

#loading haarcascade classifier in order to use its features.
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
 
#making a list here in order to save the largest flatten images as numpy arrays
face_data=[]

#taking an user input ,name of the person whose photo is being taken."
file_name=input("Enter the name of the person whose is being taken : ")

#Driver Code
while True:
    boolean,frame=cap.read()

    if boolean==False:
        continue
     
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    
    if len(faces)==0:
        continue
    
    faces=sorted(faces,key=lambda f:f[2]*f[3],reverse=True)
      
    for (x,y,w,h) in faces:
    
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        
        # here we crop out the region of interest
        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        
        #resizing the photo according to the requirement so that detector can work efficiently
        face_section=cv2.resize(face_section,(100,100))
        
        skip+=1
        #here skip is used so that we can store every 10th image that is captured, avoiding the other 9.
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data))

    cv2.imshow("Photo",frame)
    cv2.imshow("Region of Interest",face_section)
    
    #press 'q' , the above process will stop
    
    keypressed=cv2.waitKey(1) & 0xFF
    if keypressed==ord('q'):
        break
   
#converting face data into numpy array
face_data=np.asarray(face_data) 

#now reshaping the face data and storing the data of one image into a single row
face_data=face_data.reshape((face_data.shape[0],-1))

#now saving the data as '.npy' file
np.save(file_name+'.npy',face_data)
cap.release()
cv2.destroyAllWindows()   
