"""
DATA.ML.100-2020-2021-1 Introduction to Pattern Recognition and Machine Learning (Fall 2020)
Creator: Patricia Rodriguez Vaquero <patricia.rodriguezvaquero@tuni.fi>
Student id number: K437765
"""

import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import random
from skimage.transform import rescale, resize, downscale_local_mean
import math
import scipy.stats as stats
from scipy.stats import norm
import time

tic = time.process_time()
def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

# Test data
datadict = unpickle('C:/Users/patir/OneDrive/Documentos/cifar-10-batches-py/test_batch')

X = datadict["data"]
Y = datadict["labels"]

labeldict = unpickle('C:/Users/patir/OneDrive/Documentos/cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]

test_images = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype(np.int)

test_classes = np.array(Y)

print(f"The shape of test_images is: {test_images.shape}")
print(f"The shape of test_classes is: {test_classes.shape}")

# All training data
Training_files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
data = []
labels = []
for i in range(5):
    raw_data = unpickle('C:/Users/patir/OneDrive/Documentos/cifar-10-batches-py/' + Training_files[i])
    data.append(raw_data["data"])
    labels.append(raw_data["labels"])
train_images = np.concatenate(data)
train_classes = np.concatenate(labels)
#train_images = train_images.reshape(50000, 32, 32, 3)
train_images = train_images.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype(np.int)

print(f"The shape of train_images is: {train_images.shape}")
print(f"The shape of train_classes is: {train_classes.shape}")
print("-----------------------------------------------")


# Funciones
# TASK 1
def cifar_10_evaluate(pred, gt):
    correct = 0
    for i in range(len(pred)):
        if pred[i] == gt[i]:
            correct += 1
    return (correct / len(pred)) * 100


def class_acc(pred, gt):
    correct = 0
    for i in range(len(gt)):
        if gt[i] == pred[i]:
            correct += 1
    return correct / float(len(gt)) * 100.0

# Calculate Gaussian Probability Density Function
#def normpdf(x, mu, var):
    #denom = var*np.sqrt(2*np.pi)
    #expo = np.exp((-1/2)*((x-mu)**2/(var**2)))
    #return (1/denom)*expo


# Write a function def cifar10 color(X) that converts the original images in X (50000x32x32x3) to Xf (50000x3).
def cifar_10_color(x):
    x = np.transpose(x, [0, 3, 1, 2])
    x = np.reshape(x, [x.shape[0], -1])  # 50000,3072
    r = x[0:50000, 0:1024] #50000x1024, the same as r=x[:, 0:1024]
    g = x[0:50000, 1024:2048] #50000x1024
    b = x[0:50000, 2048:3072] #50000x1024
    r_mean = np.mean(r, axis=1, keepdims=True) #50000x1
    g_mean = np.mean(g, axis=1, keepdims=True)  # 50000x1
    b_mean = np.mean(b, axis=1, keepdims=True)  # 50000x1
    train_images = np.concatenate((r_mean, g_mean, b_mean), axis=1) #50000x3 -> 10x50000x3
    return train_images


# Write a function def cifar 10 naivebayes learn(Xf,Y) that computes the normal distribution
# parameters (mu, sigma, p) for all ten classes (mu and sigma are 10x3 and priors p is 10x1).
#TASK 1
def cifar_10_bayes_learn1(Xf, label):
    ms = []
    sg = []
    for i in range(0, 10):
        idx = np.where(label == i) # get from label-vector ith entries indexes
        #idx, _ = np.where(label==i)
        #print(Xf.shape)
        #print(idx)
        class_i = Xf[idx] # get data according to gotten indexes # 5000x3
        #class_i = Xf[idx,:]
        # calculate mean of each component
        mur = np.mean(class_i[:, 0]) # class_i[:,0]= 5000x1
        mug = np.mean(class_i[:, 1])
        mub = np.mean(class_i[:, 2])
        ms.append([mur, mug, mub])

        # calculate sigma squared of each component
        sg_r = np.std(class_i[:, 0])
        sg_g = np.std(class_i[:, 1])
        sg_b = np.std(class_i[:, 2])
        sg.append([sg_r, sg_g, sg_b])
    p = len(class_i) / len(Xf) # p: prior probability
    return [np.array(ms), np.array(sg), p]


# Write function def cifar10 classifier naivebayes(x,mu,sigma,p) that returns the Bayesian optimal class c
# for the sample x.
def cifar10_classifier_naivebayes(x, ms, gs, p): #x: 1x3
    #ms = 10x3, gs=10x3
    prob = np.zeros(10)
    for i in range(0, 10):
        pr = stats.norm.pdf(x[0], ms[i, 0], gs[i, 0])
        pg = stats.norm.pdf(x[1], ms[i, 1], gs[i, 1])
        pb = stats.norm.pdf(x[2], ms[i, 2], gs[i, 2])
        prob[i] = pr * pg * pb * p
    cls = np.argmax(prob)
    return cls
    #mx = max(prob.values())
    #cl = [c for c, nd in prob.items() if mx == nd]
    #return cl[0]

#TASK 2
def cifar_10_bayes_learn2(Xf, label):
    ms = []
    cov = []
    for i in range(0, 10):
        idx = np.where(label == i)  # get from label-vector ith entries indexes
        # idx, _ = np.where(label==i)
        # print(Xf.shape)
        # print(idx)
        class_i = Xf[idx]  # get data according to gotten indexes # 5000x3
        # class_i = Xf[idx,:]
        # calculate mean of each component
        mur = np.mean(class_i[:, 0])  # class_i[:,0]= 5000x1
        mug = np.mean(class_i[:, 1])
        mub = np.mean(class_i[:, 2])
        ms.append([mur, mug, mub])

        covariance = np.cov(class_i.T)
        cov.append(covariance)
    p = len(class_i) / len(Xf)  # p: prior probability
    return [np.array(ms), np.array(cov), p]

# TASK 3
def cifar_10_bayes_learn3(Xf, label):
    ms = []
    cov = []
    for i in range(0, 10):
        idx = np.where(label == i)  # get from label-vector ith entries indexes
        # idx, _ = np.where(label==i)
        # print(Xf.shape)
        # print(idx)
        class_i = Xf[idx]  # get data according to gotten indexes # 5000x3
        # class_i = Xf[idx,:]
        # calculate mean of each component
        #mur = np.mean(class_i[:, 0])  # class_i[:,0]= 5000x1
        #mug = np.mean(class_i[:, 1])
        #mub = np.mean(class_i[:, 2])
        mu = np.mean(class_i, axis=0)
        #ms.append([mur, mug, mub])
        ms.append(mu)
        covariance = np.cov(class_i.T)
        cov.append(covariance)
        #print("----------------------------------------------------------------")
    p = len(class_i) / len(Xf)  # p: prior probability
    return [np.array(ms), np.array(cov), p]

def cifar10_classifier_bayes2(x, ms, cov, p): #x: 1x3, cov:10x3x3
    prob = np.zeros(10)
    for j in range(0, 10):
        p_total = stats.multivariate_normal.pdf(x, ms[j, :], cov[j, :, :]) #cov size 1x3x3
        prob[j] = p_total*p
    cls = np.argmax(prob)
    return cls


def cifar10_classifier_bayes3(x, ms, cov, p): #x: 1x3, cov:10x3x3
    prob = np.zeros([10000, 10])
    for i in range(0, 10):
        p_total = stats.multivariate_normal.logpdf(x, ms[i, :], cov[i, :, :]) #cov size 1x3x3
        #print(f"The shape of p_total is {p_total.shape}")
        prob[:, i] = p_total*p
    cls = np.argmax(prob, axis=1)
    return cls


def cifar10_2x2_color(images, size=(2, 2)):
    reshaped_image = np.reshape(images, [images.shape[0], 3, 32, 32]).transpose([0, 2, 3, 1])
    reshaped_image = np.array(reshaped_image, dtype='uint8')
    out = np.zeros([images.shape[0], size[0], size[1], 3])
    for i in range(images.shape[0]):
        out[i, :, :, :] = cv2.resize(reshaped_image[i, :, :, :], size)
    out = np.transpose(out, [0, 3, 1, 2])
    out = np.reshape(out, (out.shape[0], -1))
    return np.array(out, dtype=np.int)

# Getting the Accuracy and Computing time
# TASK 1
Xf1 = cifar_10_color(train_images)
#print(f"The shape of Xf is: {Xf.shape}")
[ms, gs, p] = cifar_10_bayes_learn1(Xf1, train_classes)

#print(ms.shape, gs.shape): 10x3 each of them
test_images1 = cifar_10_color(test_images) # 10000x3
#print(f"The shape of test_images is: {test_images.shape}")

pred_cls1 = np.zeros(10000)
for x in range(0, 10000):
    pred_cls1[x] = cifar10_classifier_naivebayes(test_images1[x], ms, gs, p)
    #print(pred_cls)

print(f"The accuracy for Naivebayes Classifier is: {cifar_10_evaluate(pred_cls1, test_classes)}")

toc = time.process_time()
print(f"The computing time for Task 1 is: {toc - tic}")
print("-----------------------------------------------")


#TASK 2
Xf2 = cifar_10_color(train_images)
#print(f"The shape of Xf is: {Xf.shape}")
[ms, cov, p] = cifar_10_bayes_learn2(Xf2, train_classes)

#print(f"The covariance shape is: {cov.shape}") #10x3x3 each of them

test_images2 = cifar_10_color(test_images) # 10000x3
#print(f"The shape of test_images is: {test_images.shape}")

pred_cls2 = np.zeros(10000)
for x in range(0, 10000):
    pred_cls2[x] = cifar10_classifier_bayes2(test_images2[x], ms, cov, p)
    #print(pred_cls2)

print(f"The accuracy for Bayes Classifier is: {cifar_10_evaluate(pred_cls2, test_classes)}")

toc = time.process_time()
print(f"The computing time for Task 2 is: {toc - tic}")
print("-----------------------------------------------")


# Task 3
# PLOT A GRAPH
accuracy_metric = []
for i in range(0, 6):
    size = 2**i
    Xf3 = cifar10_2x2_color(train_images, size=(size, size))
    [ms, cov, p] = cifar_10_bayes_learn3(Xf3, train_classes)
    imagenes_test = cifar10_2x2_color(test_images, size=(size, size))  # 10000x3
    pred_cls3 = cifar10_classifier_bayes3(imagenes_test, ms, cov, p)
    accuracy_result = cifar_10_evaluate(pred_cls3, test_classes)
    accuracy_metric.append(accuracy_result)
    print(f"The Bayes accuracy for size {size} x {size} is: {accuracy_metric[i]}")

#print(f"List of Accuracy Metrics: {accuracy_metric}")

# GRAPH
x = ["1x1", "2x2", "4x4", "8x8", "16x16", "32x32"]
y = accuracy_metric
plt.plot(x, y, "o:", color='green')
plt.xlabel('Images Shape')
plt.ylabel('Accuracy')
plt.show()

toc = time.process_time()
print(f"The computing time for Task 3 (all sizes) is: {toc - tic}")