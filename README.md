# The Color Blind Accessibility Checker

The Color Blind Accessibility Checker (CBA Checker)  focuses on creating a color blind accessibility checker for shop signs: running word detection, image simulation, and color comparison methods to determine if sign images are readable for people with color blindness.  It also aims to help raise awareness for accessibility to the visually impaired.

This project hosts the final code for the CBA Checker Research Paper:
```
The Color Blind Accessibility Checker for Local Shop Signs in Natural Settings
Michelle Li (2020)
```
## Installation
1. Please download the [Total-Text Dataset](https://github.com/cs-chan/Total-Text-Dataset) and change the "Test" folder (should contain 300 pictures) to be called "Signs". Then place the newly named "Signs" folder into the same folder as the rest of the code from this project.

2. The dependencies needed to run this project include [numpy](https://numpy.org/), [cv2](https://opencv.org/), csv, os, and [sklearn.cluster's KMeans module](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html).

Other notes on installation:
1. For the purposes of this project, the use of [Convolutional Character Networks](https://github.com/MalongTech/research-charnet) (CharNet), has already been applied to each of the images in "Signs." The results are stored in "OutputSigns," and are ready for use.

2. [CIEDE2000](https://github.com/lovro-i/CIEDE2000) has also been implemented with minor changes. It is stored in this project to make implementing this project easier but all credits remain with the owner.
 

## Usage

The Color Blindness Accessibility Checker (CBA Checker) is run through the "Final Code" file. 

For increased computational power and thereby faster processing speed, it is recommended to run this using [Google Colab](colab.research.google.com).

## An Example of the Results

![Increase Accessibility for Deuteranopia](https://user-images.githubusercontent.com/53533879/93836721-805a6f80-fc38-11ea-80ae-babdd010fca0.jpg)
