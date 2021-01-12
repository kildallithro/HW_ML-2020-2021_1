import numpy as np
import pandas as pd
from PIL import Image
import os
import random
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt


def readp2pgm(name):    # 读取图片
    with open(name) as f:
        lines = f.readlines()

    for l in list(lines):
        if l[0] == '#':
            lines.remove(l)

    assert lines[0].strip() == 'P2'

    data = []
    for line in lines[1:]:
        data.extend([int(c) for c in line.split()])

    return np.array(data[3:])

def readpgm(name):    # 读取图片
    with open(name,'rb') as f:
        lines = f.readlines()

    # Ignores commented lines
    for l in list(lines):
        if l[0] == '#':
            lines.remove(l)

    data = []
    # Makes sure it is ASCII format (P2)
    if  lines[0].strip() == b'P2':
        data = readp2pgm(name)
        return data
    else:
        img = Image.open(name)
        img = img.resize((128, 120))
        data = np.array(img).reshape(1, img.size[0] * img.size[1])
        return data[0]
# print(readpgm('./faces_4/an2i/an2i_up_sad_sunglasses_4.pgm').shape)


def loaddata(path):
    imagedata = []
    label = {}
    files = os.listdir(path)
    la = 0
    i = 0
    for file_paths in files:
        file_path = os.path.join(path,file_paths)

        images_file = os.listdir(file_path)

        for image_file in images_file:
            image = os.path.join(file_path,image_file)
            # img = Image.open(image)
            img = readpgm(image)
            img = img.tolist()
            imagedata.append(img)
            lab = image_file.split('_')
            label[i] = lab[2]
            i += 1
        la += 1
    label_num = {}
    labels = list(label.values())
    ima_label = list(set(labels))
    for i in range(len(labels)):
        for j in range(len(ima_label)):
            if labels[i] == ima_label[j]:
                label_num[i] = j
                break

    labels_num = list(label_num.values())
    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(imagedata)
    random.seed(randnum)
    random.shuffle(labels)

    return imagedata, labels_num
# print(loaddata('./faces_4'))

# 数据加载
X, y = loaddata('./faces_4')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# 训练模型
clf = svm.SVC(decision_function_shape='ovo')
# clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovo')
clf.fit(X_train, y_train)

# train data
print('训练集的score为: ', clf.score(X_train, y_train))
# y_train_hat = clf.predict(X_train)
# precision_train = sum(y_train_hat == y_train) / len(y_train)
# print('训练集的精确度为: ', precision_train)

# Test data
print('测试集的score为: ', clf.score(X_test, y_test))
# y_test_hat = clf.predict(X_test)
# precision_test = sum(y_test_hat == y_test) / len(y_test)
# print('测试集的精确度为: ', precision_test)
