# Introduction
Face recognition is a crucial security application. Through this project,  a very basic form of face recognition has been implemented using the Haar Cascades Classifier, openCV & K-Nearest Neighbors Algorithm.

#KNN
The k-nearest neighbor classifier fundamentally relies on a distance metric. The better that metric reflects label similarity, the better the classified will be. The most common choice is the Minkowski distance 

				dist(x,z)=(d∑r=1|xr−zr|p)1/p.

# Face-Recognition 
Face Recognition using custom KNN algorithm (computed without using NumPy) and open cv for python.

## Technology Stack
Python - The whole code has been written in Python
cv2 -  cv2 is the OpenCV module and is used here for reading & writing images & also to input a video stream
Algorithm - KNN
Classifier - Haar Cascades


## Breakdown of the code for KNN classifier
    1. Importing libraries
    2. Create some data for classification
    3. Write the kNN workflow
    4. Finally, run knn on the data and observe results
## Dependencies
    Python 2.7 and OpenCv

## How it works!

* Clone the Repo!
* Run face_detection.py script to capture images of the person using delfaut camera
	- Enter the 'name' of the peron
	- Let the camera capture images in diferent angles
	- Enter 'q' to quit.
	- Repeat the Process for different Persons
	- This saves the images in NumPy array format name.npy
* Run Face_recog_CustomKNN.py to recognise the faces detected
	- This scrip takes the numpy array saved and matches with faces
	- If matched it displays by assigning 'name; to the face.
* Run Face_RecCus_CosineSM.py to recognise the faces detected using KNN where destiance is calculated using Cosine similiraity instead of Euclidean distance

