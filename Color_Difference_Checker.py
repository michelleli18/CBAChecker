import numpy as np
import cv2
import csv
from sklearn.cluster import KMeans
from skimage import color
from ciede2000 import ciede
import os


def charNet(img, imgName, colorBlindImg):
    # the following pre-processing steps are to extract the returned values from CharNet's word detection
    # print("OutputSigns/{}.npy".format(imgName))
    charNetResults = np.load("OutputSigns/{}.npy".format(imgName))
    if charNetResults.size == 0:
        return "No words detected"
    charNetResults = charNetResults[0]

    boundingBoxCoordinates = []
    i = 0
    while i in range(0, len(charNetResults)-2):
        # puts the charNetResults into the coordinates
        coordinate = (charNetResults[i], charNetResults[i+1])
        i = i+2
        boundingBoxCoordinates.append(coordinate)
    # print(boundingBoxCoordinates)

    imgCopy = img.copy()
    j=0
    while j in range(0, len(boundingBoxCoordinates)-2):
        # draws rectangles for original image
        cv2.rectangle(imgCopy, boundingBoxCoordinates[j], boundingBoxCoordinates[j+2], color = (0,255,0), thickness = 3)
        # create a black image same size as original
        hgt, wid, dep = img.shape
        mask = np.zeros((hgt, wid), dtype=np.uint8)
        # create a white rectangle on mask based off boundingBoxCoordinates
        cv2.rectangle(mask, boundingBoxCoordinates[j], boundingBoxCoordinates[j + 2], color=(255, 255, 255), thickness=-1)
        # passes the colorblind image and each character mask into the contrast checker
        # this way, the results from the charNet will be used to test the colors from the simulation images
        differenceResult = differenceChecker(colorBlindImg, mask)
        if differenceResult == False:
            return False
        j = j + 4
    return True

def differenceChecker(colorBlindImg, mask):
    # creates an image that only has one character
    characterImg = cv2.bitwise_and(colorBlindImg, colorBlindImg, mask=mask)
    rgbCharacterImg = cv2.cvtColor(characterImg, cv2.COLOR_BGR2RGB)
    # finds the three most dominant colors in the character image
    # since we're using black images as masks, one of them will be black
    dominantColors = findColors(rgbCharacterImg)
    # the first one is black
    # the two most common colors left will be the text and it's background
    color1, color2 = dominantColors[1:]
    # change rgb to lab
    color1Lab = rgbToLab(color1)
    color2Lab = rgbToLab(color2)
    # pass lab to delta e
    # return true or false for distance checking
    distanceResult = compareDistance(color1Lab, color2Lab)
    return distanceResult


def findColors(img):
    # separate colors to two clusters
    # they represent the two different types of color groups
    num_clusters = 3
    clusters = KMeans(n_clusters=num_clusters)
    hgt, wid, dep = img.shape
    # reshapes to match the clusters.fit parameter requirements
    # this reshapes it to be just a column of rgb values, no order
    colorsOnly = np.reshape(img, (hgt * wid, -1))
    clusters.fit(colorsOnly)
    # cluster centers are the most common colors in each group
    # print("cluster centers are:")
    # print(clusters.cluster_centers_.astype(int))
    dominantColors = clusters.cluster_centers_.astype(int)
    return dominantColors

def rgbToLab(rgb):
    #turn rgb to sRGB
    RsRGB = rgb[0]/255.0
    GsRGB = rgb[1]/255.0
    BsRGB = rgb[2]/255.0
    sRGB = (RsRGB, GsRGB, BsRGB)
    # the imported color module is used for rgb2lab
    # this requires the format to have three []s
    # print("sRGB is:", sRGB)
    lab = color.rgb2lab([[[sRGB]]])
    return lab

def compareDistance(lab1, lab2):
    # parses the lab color's formatting
    # originally has three []
    lab1 = lab1[0][0][0]
    lab2 = lab2[0][0][0]
    # find deltaE distance
    distance = ciede(lab1, lab2)
    # the threshold for colors to still be differentiated
    # according to Delta E 101 (Schuessler, 2015)
    if distance <= 11:
        return False
    else:
        return True

