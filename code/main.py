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


def build_vocabulary(image_paths, dico):
    dico = []
    sift = cv2.SIFT_create()

    for path in image_paths:
        img = cv2.imread(path)

        kp, des = sift.detectAndCompute(img, None)
        for d in des:
            dico.append(d)

    k = 60

    x = 3
    batch_size = int(1253 * x)
    vocab = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, verbose=1).fit(dico)

    vocab.verbose = False

    histo_list = []
    for path in image_paths:
        img = cv2.imread(path)
        kp, des = sift.detectAndCompute(img, None)

        histo = np.zeros(k)
        nkp = np.size(kp)

        for d in des:
            idx = vocab.predict([d])
            histo[idx] += 1 / nkp  # Because we need normalized histograms, I prefere to add 1/nkp

        histo_list.append(histo)

    return histo_list


def get_tiny_images(image_paths):
    height = 16
    width = 16

    tiny_images = np.zeros((len(image_paths), width * height))

    for i, image_data in enumerate(image_paths):
        image = Image.open(image_data)
        image_re = np.asarray(image.resize((width, height), Image.ANTIALIAS), dtype='float32').flatten()
        image_nm = (image_re - np.mean(image_re)) / np.std(image_re)
        tiny_images[i, :] = image_nm

    return tiny_images


if __name__ == "__main__":

    dico = []
    test_sample_counts = dict()
    test_labels_accurs = dict()

    train_labels = []
    test_labels = []

    train_paths = []
    test_paths = []
    train_image_feats = []
    test_image_feats = []
    pred_labels = []

    #gatherig the data pets

    train_root = 'SceneDataset/train'
    test_root = 'SceneDataset/test'

    for dirname in os.listdir(train_root):
        dir_path = train_root + "/" + dirname
        for fname in os.listdir(dir_path):
            train_labels.append(dirname)
            impath = dir_path + "/" + fname
            train_paths.append(impath)

    for dirname in os.listdir(test_root):
        dir_path = test_root + "/" + dirname
        test_sample_counts[dirname] = len(os.listdir(dir_path))
        for fname in os.listdir(dir_path):
            test_labels.append(dirname)
            impath = dir_path + "/" + fname
            test_paths.append(impath)

    feature = sys.argv[1]
    classify = sys.argv[2]

    print("FEATURE : %s ClASSIFY : %s " % (feature, classify))

    # tiny image
    if (feature == "tiny"):
        train_image_feats = get_tiny_images(train_paths)
        test_image_feats = get_tiny_images(test_paths)

    # BAG OF WORDS

    if (feature == "bow"):
        try:
            train_image_feats = np.load("hist.npy")
        except:
            train_image_feats = build_vocabulary(train_paths, dico)
            np.save("hist.npy", train_image_feats)

        try:
            test_image_feats = np.load("test_feats.npy")
        except:
            test_image_feats = build_vocabulary(test_paths, dico)
            np.save("test_feats.npy", test_image_feats)

    if (classify == "knn"):
        knn = KNeighborsClassifier(n_neighbors=5)
        c = knn.fit(train_image_feats, train_labels)

        pred_labels = knn.predict(test_image_feats)

    if (classify == "svm"):
        C = 10.0
        G = 0.01
        K = 'linear'
        clf = svm.SVC(C=C, gamma=G, kernel=K)
        c = clf.fit(train_image_feats, train_labels)
        pred_labels = clf.predict(test_image_feats)
        print("SVM parameters : C = %.1f, Gamma : %.2f Kernel : %s " % (C, G, K))

    cor_found_labels = {"Bedroom": 0, "Highway": 0, "Kitchen": 0, "LivingRoom": 0, "Mountain": 0, "Office": 0}
    match = 0
    for i in range(len(pred_labels)):
        pred = pred_labels[i]
        act = test_labels[i]
        if (pred == act):
            match += 1
            cor_found_labels[act] = cor_found_labels[act] + 1

    for i in cor_found_labels.keys():
        acc = cor_found_labels[i] / test_sample_counts[i]
        test_labels_accurs[i] = "%.2f" % acc


    print(cor_found_labels)
    print(test_sample_counts)
    print(test_labels_accurs)

    confusion_matrix(test_labels, pred_labels, labels=["Bedroom", "Highway", "Kitchen","LivingRoom","Mountain","Office"])

    disp = plot_confusion_matrix(c, test_image_feats, test_labels,
                                 display_labels=["Bedroom", "Highway", "Kitchen","LivingRoom","Mountain","Office"],
                                 cmap=plt.cm.Blues,
                                 )

    accur = match / len(pred_labels)
    print(accur)


    plt.show()


    plt.close('all')
