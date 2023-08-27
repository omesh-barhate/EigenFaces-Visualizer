# Import necessary packages.
from __future__ import print_function
import os
import sys
import cv2
import numpy as np


def readImages(path):
    print("Reading images from " + path, end = "...")


    images = []

    for filePath in sorted(os.listdir(path)):
        fileExt = os.path.splitext(filePath)[1]
        if fileExt in [".jpg", ".jpeg"]:


            imagePath = os.path.join(path, filePath)
            im = cv2.imread(imagePath)

            if im is None :
                print("image:{} not read properly".format(imagePath))
            else :

                im = np.float32(im)/255.0

                images.append(im)

                imFlip = cv2.flip(im, 1);
 
                images.append(imFlip)
    numImages = int(len(images) / 2)

    if numImages == 0 :
        print("No images found")
        sys.exit(0)

    print(str(numImages) + " files read.")
    return images


def createDataMatrix(images):
    print("Creating data matrix", end = " ... ")
    ''' 
	Allocate space for all images in one data matrix.
	The size of the data matrix is
	( w  * h  * 3, numImages )
	where,
	w = width of an image in the dataset.
	h = height of an image in the dataset.
	3 is for the 3 color channels.
	'''
    numImages = len(images)
    sz = images[0].shape

    data = np.zeros((numImages, sz[0] * sz[1] * sz[2]), dtype = np.float32)
    for i in range(0, numImages):
        image = images[i].flatten()

        data[i,:] = image

    print("DONE")
    return data


def createNewFace(*args):

    output = averageFace


    for i in range(0, NUM_EIGEN_FACES):

        '''
		OpenCV does not allow slider values to be negative. 
		So we use weight = sliderValue - MAX_SLIDER_VALUE / 2
		''' 
        sliderValues[i] = cv2.getTrackbarPos("Weight" + str(i), "Trackbars");
        weight = sliderValues[i] - MAX_SLIDER_VALUE/2

        output = np.add(output, eigenFaces[i] * weight)


    output = cv2.resize(output, (0,0), fx = 2, fy = 2)
    cv2.imshow("Result", output)

def displayEigenfaces(eigenFaces):
    for i, eigenFace in enumerate(eigenFaces):
        cv2.imshow("Eigenface " + str(i+1), eigenFace)


def rankEigenfaces(eigenValues):
    sorted_indices = np.argsort(eigenValues)[::-1]
    return sorted_indices


def resetSliderValues(*args):
    for i in range(0, NUM_EIGEN_FACES):
        cv2.setTrackbarPos("Weight" + str(i), "Trackbars", int(MAX_SLIDER_VALUE/2));
    createNewFace()
    
if __name__ == '__main__':
    NUM_EIGEN_FACES = 202
    MAX_SLIDER_VALUE = 255
    dirName = "data"
    images = readImages(dirName)
    sz = images[0].shape
    data = createDataMatrix(images)
    print("Calculating PCA ", end="...")
    mean, eigenVectors = cv2.PCACompute(data, mean=None, maxComponents=NUM_EIGEN_FACES)
    print("DONE")
    averageFace = mean.reshape(sz)
    eigenFaces = []
    for eigenVector in eigenVectors:
        eigenFace = eigenVector.reshape(sz)
        eigenFaces.append(eigenFace)
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Average", cv2.WINDOW_NORMAL)
    output = cv2.resize(averageFace, (0,0), fx=2, fy=2)
    cv2.imshow("Result", output)
    cv2.imshow("Average", averageFace)
    cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
    sliderValues = []
    for i in range(0, NUM_EIGEN_FACES):
        sliderValues.append(int(MAX_SLIDER_VALUE/2))
        cv2.createTrackbar("Weight" + str(i), "Trackbars", int(MAX_SLIDER_VALUE/2), MAX_SLIDER_VALUE, createNewFace)
    cv2.setMouseCallback("Average", resetSliderValues);
    eigenValues = eigenVectors.var(axis=0)
    sorted_indices = rankEigenfaces(eigenValues)
    for i, idx in enumerate(sorted_indices):
        print("Eigenface", i+1, "- Eigenvalue:", eigenValues[idx])
    print("\nUsage:\nChange the weights using the sliders.\nMouse hover on the result window to reset sliders.\nPress q to terminate.")
    key = cv2.waitKey(0)
    if key == ord('q'):
        cv2.destroyAllWindows()

