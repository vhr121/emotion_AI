import argparse
import glob
import os
import random
import subprocess
import sys
import time
import cv2
import pandas
import statistics
import eigen_update_model
Capture = cv2.VideoCapture(0)
expression = ['happy', 'sad', 'surprise', 'neutral', 'angry']
exp_dict = {'happy': 0, 'sad': 1, 'surprise': 2, 'neutral': 3, 'angry': 4}
exp_index = [0, 1, 2, 3, 4]
samples = 20
old_expr = 'wait!'

#take the ooptions from the users in form of arguments
parser = argparse.ArgumentParser(description="Options for the emotion-based music player")  # Create parser object
parser.add_argument("--update", help="Call to grab new images and update the model accordingly",action="store_true")  # Add --update argument
parser.add_argument("--train", help="Train the model", action="store_true")
args = parser.parse_args()  # Store any given arguments in an object


#open a music player
def open_stuff(filename):  # Open the file, credit to user4815162342, on the stackoverflow link in the text above
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])

#check if a folder and file exist else create one
def check_file_exists(folder, file_check):
    if os.path.exists("%s/%s" % (folder, file_check)):
        pass
    else:
        os.makedirs("%s/%s" % (folder, file_check))

#read the image from the web camera and return the face
def read_img_roi():
    #read the frame
    ret, frame = Capture.read()
    #convert to grey image

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #convert to clahe image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)
    #identify the face using haar cascade
    face_cascade = cv2.CascadeClassifier('face_cascade/haarcascade_frontalface_default.xml')
    face_cord = face_cascade.detectMultiScale(clahe_image, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10),flags=cv2.CASCADE_SCALE_IMAGE)

    #if face is present
    if len(face_cord) == 1:
        print "found face"
        #crop the face rectangle
        for (x, y, w, h) in face_cord:
            face_img = clahe_image[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

         #show the text over image
        font = cv2.FONT_HERSHEY_SIMPLEX
        if args.update:
            cv2.putText(frame, 'training', (x, y + h + 30), font, 1.5, (255, 255, 255), 2, cv2.CV_AA)
        else:
            cv2.putText(frame, old_expr, (x, y + h + 30), font, 1.5, (255, 255, 255), 2, cv2.CV_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit(0)

        #resize the image
        face_img=cv2.resize(face_img,(350,350))
        return face_img, True

    else:
        print "no face found"
        return [], False






#make the datasets
def make_training_data(exp):
    check_file_exists('datasets', exp)
    #discard first 4 frames
    delay = 4
    #check the number of existing files
    count_files = len(glob.glob('datasets/%s/*' % exp))
    remaining_to_read=samples
    while remaining_to_read>0:
        try:
            face_image, ret = read_img_roi()
            if ret == True:
                if delay > 0:
                    delay = delay - 1

                else:
                    cv2.imwrite('datasets/%s/%s.jpeg' % (exp, count_files), face_image)
                    count_files = count_files + 1
                    remaining_to_read=remaining_to_read-1
        except:
            print "Unexpected error:", sys.exc_info()[0]
            return -1


def make_data_sets():
    for i in expression:
        print "look at the camera and be %s" % i
        for j in range(6):
            print 5 - j
            time.sleep(1)
        make_training_data(i)

if args.update:
    make_data_sets()

elif args.train:
    eigen_update_model.update(expression)


else:

        actions = {}
        df = pandas.read_excel("EmotionLinks.xlsx")  # open Excel file
        actions["sad"] = [x for x in
                          df.sad.dropna()]  # We need de dropna() when columns are uneven in length, which creates NaN values at missing places. The OS won't know what to do with these if we try to open them.
        actions["happy"] = [x for x in df.happy.dropna()]
        actions["surprise"] = [x for x in df.surprise.dropna()]
        actions["neutral"] = [x for x in df.neutral.dropna()]
        actions["angry"] = [x for x in df.angry.dropna()]

        eigen = cv2.createEigenFaceRecognizer()
        try:
            eigen.load("models/eigen/trained_eigen_emoclassifier_2.xml")
        except:
            print("no xml found. Using --update will create one.")
            exit(1)
        while(1):

            prediction = []
            conf = []
            count = 10
            while (count > 0):

                face, ret = read_img_roi()

                if ret == True:

                    pred, confidence = eigen.predict(face)
                    prediction.append(pred)
                    conf.append(confidence)
                    count = count - 1
                else:
                    print "no face found1"
            try:
                pred = statistics.mode(prediction)
            except:
                pass

            confi = statistics.mean(conf)

            print "expression is %s" % expression[pred]
            print "confidence level is %s" % str(confi)

            #use for testing
            #check_file_exists('results', expression[pred])
            #count = len(glob.glob('results/%s/*' % expression[pred]))
            #cv2.imwrite("results/%s/%s.jpeg" % (expression[pred], str(count + 1)), face)


            if old_expr == expression[pred]:
                pass
            else:
                actionlist = [x for x in actions[expression[pred]]]  # <----- get list of actions/files for detected emotion
                random.shuffle(actionlist)  # <----- Randomly shuffle the list
                open_stuff(actionlist[0])  # <----- Open the first entry in the list
                old_expr = expression[pred]

