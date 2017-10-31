import cv2
import glob
import time
import random
import numpy as np
import os

#create eigen face recogniser object
fisher = cv2.createFisherFaceRecognizer()

data = {}

#check if a file already exist if not create the file
def check_file_exists(folder, file_check):
    if os.path.exists("%s/%s" % (folder, file_check)):
        pass
    else:
        os.makedirs("%s/%s" % (folder, file_check))

#create the training and test data
def make_sets(emotions):
    data = []
    labels = []

    for emotion in emotions:
        training = glob.glob("datasets/%s/*" % emotion)
        for item in training:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            data.append(gray)
            labels.append(emotions.index(emotion))
    #combine data and lable and shuffel used for testing purpose
    bind_lst = list(zip(data,labels))
    random.shuffle(bind_lst)
    data,labels=zip(*bind_lst)

    return data,labels


#train the model using fisher face algorithm
def run_recognizer(emotions):
    data,labels = make_sets(emotions)
    #segregate the data and lables as train and test data 80% train and 20% test
    training_labels = labels[:int(len(data)*0.8)]
    training_data= data[:int(len(data)*0.8)]
    testing_data=data[int(len(data)*0.8):]
    testing_labels=labels[int(len(data)*0.8):]
    print "total data is %s"%str(len(data))
    print("training fisher face classifier")
    print("size of training set is: " + str(len(training_labels)) + " images")
    start_time = time.time()
    fisher.train(training_data, np.asarray(training_labels))
    print "it to %s seconds to train your model"%str(time.time()-start_time)
    print "testing your model "
    count=0
    index=0
    for i in testing_data:
        exp,conf=fisher.predict(i)
        if(exp==testing_labels[index]):
            count=count+1
        index=index+1

    accuracy=(count/len(testing_data))*100.0
    print "accuracy of the model is %s"%str(accuracy)

#main calling function
def update(emotions):
    run_recognizer(emotions)
    print("saving model")
    check_file_exists('models','fisher')
    count_files = len(glob.glob('models/fisher/*'))

    fisher.save("models/fisher/trained_fisher_emoclassifier_%s.xml" %count_files)
    print("model saved!")



expression = ['happy', 'sad', 'surprise', 'angry','assistance']
