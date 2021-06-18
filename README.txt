necessary imports :
import cv2
from PIL import Image
import os
import sys
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn import svm
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt

Pathway of Images to Train and Test are supposed to be :

	SceneDataSet
		Test
			Office
			Bedroom
			Highway
			Mountain
			LivingRoom
			Kitchen
		Train
			Office
			Bedroom
			Highway
			Mountain
			LivingRoom
			Kitchen

command line argument should be as follows :
	python main.py <feature_extraction_method> <classify_method> 
	
	for Tiny Image : tiny 
	for Bag of Words : bow

	for K-Nearest Neighbour : knn
	for Linear SVM : svm 

	i.e. python main.py tiny knn 

There are 2 functios beside the main.
	•  build_vocabulary
	•  get_tiny_images
	
For bag of words there are 2 files that created,hists.npy and test_feats.npy, to increase speed in further executions.
They stores the feature of train and test images.

Detailed explanations and code snippets included in report. I declared this functions first and
later used them in main. 