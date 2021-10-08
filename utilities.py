import os
import cv2
import PIL
import numpy
import numpy as np
import scipy.io
import sklearn
from PIL import Image as pil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy import asarray, savetxt
from skimage.io import imread_collection
from random import randrange
from random import randrange
from sklearn import preprocessing


def vis_gaussian(np_array):
    figure, ax = plt.subplots(2, 5, figsize=(40, 10))
    figure.tight_layout()
    for i in range(0, 5):
        '''第一标签'''
        ax[0][i].title.set_text('Label1')
        ax[0][i].plot(np_array[i, 0, :], label="Original")
        ax[0][i].legend()
        ax[0][i].plot(np_array[i, 1, :], label="Predicted")
        ax[0][i].legend()

        '''第二标签'''
        ax[1][i].title.set_text('Label2')
        ax[1][i].plot(np_array[i, 2, :], label="Original")
        ax[1][i].legend()
        ax[1][i].plot(np_array[i, 3, :], label="Predicted")
        ax[1][i].legend()
    plt.show()


def load_npy(path):
    '''normal distribution test'''
    np_array = np.load(path)
    tmp = np_array[406]
    savetxt('results/labels406_data.csv', tmp, delimiter=',')

    figure, ax = plt.subplots(2, 5, figsize=(40, 10))
    figure.tight_layout()
    for i in range(0, 5):
        index = randrange(1496)
        '''第一标签'''
        ax[0][i].title.set_text('Label1 at #' + str(index))
        ax[0][i].plot(np_array[index, 0, :], label="Original")
        ax[0][i].legend()
        ax[0][i].plot(np_array[index, 1, :], label="Predicted")
        ax[0][i].legend()

        '''第二标签'''
        ax[1][i].title.set_text('Label2 at #'+ str(index))
        ax[1][i].plot(np_array[index, 2, :], label="Original")
        ax[1][i].legend()
        ax[1][i].plot(np_array[index, 3, :], label="Predicted")
        ax[1][i].legend()
    plt.show()


def load_images(folder):
    # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    # img = cv2.imread("data/original/Templates/Mw_1.80_Angle_0_Length_0.01.jpg")  # Read image
    # crop_img = img[50:580, 115:780]
    # resize_img = cv2.resize(crop_img, (400, 400))  # Resize image
    # cv2.imshow("output", resize_img)  # Show image
    # cv2.waitKey(0)
    labels = scipy.io.loadmat('data/original/label_guass.mat')['label']
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        crop_img = img[50:580, 115:780]
        resize_img = cv2.resize(crop_img, (32, 32))  # Resize image
        if img is not None:
            images.append(resize_img)
    mat = numpy.array(images)

    array1_shuffled, array2_shuffled = sklearn.utils.shuffle(mat, labels)
    train_data, test_data = array1_shuffled[:5984, ...], array1_shuffled[5984:, ...]
    train_labels, test_labels = array2_shuffled[:5984, ...], array2_shuffled[5984:, ...]

    np.save('data/cleaned_temp_32/train_data.npy', train_data)
    np.save('data/cleaned_temp_32/test_data.npy', test_data)
    np.save('data/cleaned_temp_32/train_labels.npy', train_labels)
    np.save('data/cleaned_temp_32/test_labels.npy', test_labels)
    print('finished')


def load_images_v2():
    folder = 'data/original/Templates'
    # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    # img = cv2.imread("data/original/Templates/Mw_1.80_Angle_0_Length_0.01.jpg")  # Read image
    # crop_img = img[50:580, 115:780]
    # resize_img = cv2.resize(crop_img, (400, 400))  # Resize image
    # cv2.imshow("output", resize_img)  # Show image
    # cv2.waitKey(0)
    labels = scipy.io.loadmat('data/original/label_guass.mat')['label']
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        crop_img = img[50:580, 115:780]
        resize_img = cv2.resize(crop_img, (128, 128))  # Resize image
        if img is not None:
            images.append(resize_img)
    mat = numpy.array(images)

    array1_shuffled, array2_shuffled = sklearn.utils.shuffle(mat, labels)
    train_data, test_data = array1_shuffled[:5984, ...], array1_shuffled[5984:, ...]
    train_labels, test_labels = array2_shuffled[:5984, ...], array2_shuffled[5984:, ...]

    np.save('data/cleaned_temp_128/train_data.npy', train_data)
    np.save('data/cleaned_temp_128/test_data.npy', test_data)
    np.save('data/cleaned_temp_128/train_labels.npy', train_labels)
    np.save('data/cleaned_temp_128/test_labels.npy', test_labels)
    print('finished')
