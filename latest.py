import tensorflow as tf

import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

import cv2

from sklearn.metrics import confusion_matrix
import itertools

#from PIL import Image 

model = Sequential()

model = load_model("fer-sad+affectnetdsa.h5")


#------------------------------
#function for drawing bar chart for emotion preditions
def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))
    
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    
    plt.show()


emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
emotionCount = [0,0,0,0,0,0,0]
frontFaces = np.empty((0,4),int)
profileFaces = np.empty((0,4),int)
    
def drawRekt(img,faces,side):
    for(x,y,w,h) in faces:
        if side == 'f':
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) #draw rectangle to main image
        elif side=='l':
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                
def predictEmotion(img,faces):
    global emotionCount
    for (x,y,w,h) in faces:
        detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
        detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
    		      	
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
    		
        img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
    		
        predictions = model.predict(img_pixels)
        #write emotion text above rectangle
        max_index = np.argmax(predictions[0])
        #print(predictions)
        emotionCount[max_index] = emotionCount[max_index] + 1
        emotion = emotions[max_index]
        cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
def resolveConflicts():
    global frontFaces
    global profileFaces 
    newFrontFaces = np.empty((0,4),int)
    newProfileFaces = np.empty((0,4),int)
    
    for pface in profileFaces:
        face_ind = np.where(profileFaces == pface)
        for face in frontFaces:
            if face[0] in range(pface[0],pface[0]+pface[2]) and (face[0] + face[2]) in range(pface[0],pface[0]+pface[2]):
                print("case 1")
                #print("Before"+str(len(frontFaces)))
                newProfileFaces = np.delete(profileFaces,face_ind,axis=0) 
                #print("NFF Shape:")
                #print(np.shape(newFrontFaces))
                #print(newFrontFaces)
                profileFaces = newProfileFaces
                #print(frontFaces)
                #print("After"+str(len(frontFaces)))
            elif face[0] in range(pface[0],pface[0]+pface[2]) and (face[0] + face[2]/2) in range(pface[0],pface[0]+pface[2]): 
                print("case 2")
                #print(face)
                #print("Before"+str(len(frontFaces)))
                #print("NFF Shape:")
                #print(np.shape(newFrontFaces))
                #print(newFrontFaces)
                newProfileFaces = np.delete(profileFaces,face_ind,axis=0) 
                profileFaces = newProfileFaces
                #print(frontFaces)
                #print("After"+str(len(frontFaces)))
            elif (face[0] + face[2]/2) in range(pface[0],pface[0]+pface[2]) and (face[0] + face[2]) in range(pface[0],pface[0]+pface[2]):                   
                #print("case 3")
                #print(face)
                #print("Before"+str(len(frontFaces)))
                #print("NFF Shape:")
                #print(np.shape(newFrontFaces))
                newProfileFaces = np.delete(profileFaces,face_ind,axis=0) 
                #print(newFrontFaces)
                profileFaces = newProfileFaces
                #print(frontFaces)
                #print("After"+str(len(frontFaces)))
        #for face in newProfileFaces:
            #if (face[0] in range(pface[0],pface[0]+pface[2])) and (pface[0]+pface[2] in range(face[0],face[0]+face[2])):
                  #newProfileFaces
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
profile_face = cv2.CascadeClassifier('haarcascades/haarcascade_profileface.xml')


#Input real-time image
#cap = cv2.VideoCapture(0)
#Input Video File
cap = cv2.VideoCapture('bohemian.mp4')
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

while(True):
    
    emotionCount = [0,0,0,0,0,0,0]
    open('file.txt', 'w').close()
    ret, img = cap.read()
    if ret:
        cv2.imwrite("girls_like_you/frame%d.jpg" ,img) #save image
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mirror = cv2.flip(img,1)
    #mirror.show()
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    frontFaces = faces
    
    rightProfileFaces = np.empty((0,4),int)
    leftProfileFaces = np.empty((0,4),int)
    
    faces = profile_face.detectMultiScale(gray, 1.3, 5)
    if len(faces)!=0:
        leftProfileFaces = faces
    
    faces = profile_face.detectMultiScale(mirror, 1.3, 5)
    if len(faces)!=0:
        #print("here")
        #print(faces)
        for face in faces:  
            face[0] = mirror.shape[1] - face[0] - face[2]  
        #print(faces)
        rightProfileFaces = faces
    
    profileFaces = np.concatenate((leftProfileFaces,rightProfileFaces),axis=0)
    
    #print("f:")
    #print(frontFaces)
    #print("p:")
    #print(profileFaces)
    
    resolveConflicts()
    
    drawRekt(img,frontFaces,'f')
    drawRekt(img,profileFaces,'p')
	
    predictEmotion(img,frontFaces)
    predictEmotion(img,profileFaces)

    cv2.putText(img,"Profile : "+str(len(profileFaces)), (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(img,"Front : "+str(len(frontFaces)), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    totalFaces = len(profileFaces)+len(frontFaces)
    
    if totalFaces!=0:
        for q in range(7):
            emotionCount[q] = (emotionCount[q]/totalFaces)*100
    
    with open('realTimeEmotionData.txt', 'w') as f:
        for countE in range(0,7):
            if countE == 6:
                f.write("%f" % emotionCount[countE])
            else:
                f.write("%f\n" % emotionCount[countE])
    
    cv2.imshow('img',img)
    #cv2.imshow('mirror',mirror)
    if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
        break
    
    ret, img = cap.read()

    
#kill open cv things		
cap.release()
cv2.destroyAllWindows()



'''

#Input static image
img = cv2.imread('test_emo_3.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #transform image to gray scale

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#detectFaces(faces,'f',tempPreds)


faces = profile_face.detectMultiScale(gray, 1.3, 5)
#detectFaces(faces,'l',tempPreds)
    

mirror = cv2.flip(img,1)
faces = profile_face.detectMultiScale(mirror, 1.3, 5)
#detectFaces(faces,'r',tempPreds)





#------------------------------
#variables
num_classes = 7 #angry, disgust, fear, happy, sad, surprise, neutral
batch_size = 256
epochs = 100
#------------------------------
#read kaggle facial expression recognition challenge dataset (fer2013.csv)
#https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge

with open("dataset/fer2013-sad+affectnetdsa.csv") as f:
    content = f.readlines()

lines = np.array(content)

num_of_instances = lines.size
print("number of instances: ",num_of_instances)
print("instance length: ",len(lines[1].split(",")[1].split(" ")))

#------------------------------
#initialize trainset and test set
x_train, y_train, x_test, y_test = [], [], [], []
x_val, y_val = [], []
#------------------------------
#transfer train and test set data
for i in range(1,num_of_instances):
    try:
        emotion, img, usage = lines[i].split(",")
          
        val = img.split(" ")
            
        pixels = np.array(val, 'float32')
        
        emotion = keras.utils.to_categorical(emotion, num_classes)
    
        if 'Training' in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)
        elif 'PrivateTest' in usage:
            y_val.append(emotion)
            x_val.append(pixels)
    except:
        print("",end="")

#------------------------------
#data transformation for train and test sets
x_train = np.array(x_train, 'float32')
y_train = np.array(y_train, 'float32')
x_val = np.array(x_val, 'float32')
y_val = np.array(y_val, 'float32')
x_test = np.array(x_test, 'float32')
y_test = np.array(y_test, 'float32')

x_train /= 255 #normalize inputs between [0, 1]
x_val /= 255
x_test /= 255

x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_train = x_train.astype('float32')
x_val = x_val.reshape(x_val.shape[0], 48, 48, 1)
x_val = x_val.astype('float32')
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
x_test = x_test.astype('float32')

print(x_train.shape[0], 'train samples')
print(x_val.shape[0], 'validation samples')
print(x_test.shape[0], 'test samples')
#------------------------------
#construct CNN structure
model = Sequential()

#1st convolution layer
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

#2nd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

model.add(Flatten())

#fully connected neural networks
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

#------------------------------
#batch process
gen = ImageDataGenerator(featurewise_center=False,
                         featurewise_std_normalization=False,
                         rotation_range=10,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=.1,
                         horizontal_flip=True
                         )

train_generator = gen.flow(x_train,
                           y_train,
                           batch_size = batch_size
                           )

#------------------------------
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy']
              )

history = model.fit_generator(train_generator, 
                              validation_data = (x_val, y_val), 
                              steps_per_epoch = batch_size, 
                              epochs = epochs
                              )

#------------------------------
#save and load model
model.save("emotion-92-57-model.h5")
model = load_model("fer-sad+affectnetdsa.h5")

#------------------------------
#plot loss, acc w.r.t epochs
print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#------------------------------
#overall evaluation
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', 100*score[1])
y_pred = model.predict(x_test, verbose = 1)

#------------------------------
#create confusion matrix
y_true = y_test.argmax(axis=1)
y_pred = y_pred.argmax(axis=1)
cm = confusion_matrix(y_true, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
title='Confusion matrix'

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(title)
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)
fmt = '0.2f'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()
#------------------------------
#make prediction for custom image out of test set
#capture frontal face of the image

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
profile_face = cv2.CascadeClassifier('haarcascades/haarcascade_profileface.xml')
img = cv2.imread('test_emo_3.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #transform image to gray scale

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
ds
for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
        detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
    
        x = image.img_to_array(detected_face)
        x = np.expand_dims(x, axis = 0)
        x /= 255
        custom = model.predict(x)
        emotion_analysis(custom[0])
        
        x = np.array(x, 'float32')
        x = x.reshape([48, 48]);
        
        plt.gray()
        plt.imshow(x)
        plt.show()

#cv2.imshow('img', img)
#------------------------------
   
'''     
