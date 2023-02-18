from sklearn.preprocessing import LabelEncoder
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from skimage.feature import greycomatrix, greycoprops
from sklearn import metrics
import warnings

warnings.simplefilter('ignore')


dataset = pd.read_csv('features.csv')
print(dataset)
print("shape", dataset.shape)


print(dataset.info())
print(dataset.isnull().sum())

# # Encode labels
le = LabelEncoder()
lable = le.fit_transform(dataset.name)
print(lable)

dataset["outcome"]=lable
print(dataset)


print("outcome0")
print(dataset[dataset.outcome==0].head())
print("outcome1")
print(dataset[dataset.outcome==1].head())

x=dataset.drop(['name',"outcome"],axis='columns')
y=dataset.outcome

print(x)
print(y)


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=2)

# Standardize features by removing the mean and scaling to unit variance.
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# print('x.shape,x_test.shape,x_train.shape')
# print(x.shape,x_test.shape,x_train.shape)


knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(x_train, y_train)
print('fitting done')
  # Make predictions on the test data
y_pred = knn.predict(x_test)
print("prediction done")
print(y_pred)
print(x_test.shape)
# y_pred1=knn.predict([[10.0525501253133,1.02940476190476,0.751117144376173,0.0890507539186577,0.996219488809826]])
# y_pred2 = knn.predict([[1.85501879699248,0.695895989974937,0.730095391626096,0.0621683302679788,0.999261838330219]])
# print(y_pred1)
# print(y_pred2)

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
# accuracy = confusion_matrix[]
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# accuracy1 = metrics.accuracy_score(y_test, y_pred)
# print(accuracy1)

# cam = cv2.VideoCapture(0)
# classifier = cv2.CascadeClassifier('C:/Users/sethy/PycharmProjects/faceRecognition/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
# while True:
#     ret, fr = cam.read()
#     if ret == True:
#             gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
#             face_coordinates = classifier.detectMultiScale(gray, 1.3, 5)

#             for (x, y, w, h) in face_coordinates:
#                 fc = fr[y:y + h, x:x + w, :]
#                 r = cv2.resize(fc, (100, 50)).flatten().reshape(1,-1)
# # #             # GLCM
#                 gCoMat = greycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
#                 contrast = greycoprops(gCoMat, prop='contrast')
#                 dissimilarity = greycoprops(gCoMat, prop='dissimilarity')
#                 homogeneity = greycoprops(gCoMat, prop='homogeneity')
#                 energy = greycoprops(gCoMat, prop='energy')
#                 correlation = greycoprops(gCoMat, prop='correlation')

#                 contrast_features = np.reshape(np.array(contrast).ravel(), (1, len(np.array(contrast).ravel())))
#                 dissimilarity_features = np.reshape(np.array(dissimilarity).ravel(), (1, len(np.array(dissimilarity).ravel())))
#                 homogeneity_features = np.reshape(np.array(homogeneity).ravel(), (1, len(np.array(homogeneity).ravel())))
#                 energy_features = np.reshape(np.array(energy).ravel(), (1, len(np.array(energy).ravel())))
#                 correlation_features = np.reshape(np.array(correlation).ravel(), (1, len(np.array(correlation).ravel())))
            
#                 features = np.concatenate((contrast_features, dissimilarity_features, homogeneity_features, energy_features,correlation_features), axis=1)
                
#                 cv2.putText(lable, y_pred[0].all, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
#                 cv2.rectangle(features, (x, y), (x + w, y + w), (0, 0, 255), 2)
#                 cv2.imshow('livetime face recognition', fr)
                # # Calculate the accuracy of the model


# accuracy = accuracy_score(y_test, y_pred)
# print('Accuracy:', accuracy)

# # classifier = cv2.CascadeClassifier('C:/Users/sethy/PycharmProjects/faceRecognition/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
# # # # # Face Recognition using KNN
# # cam = cv2.VideoCapture(0)
# # while True:
# #     ret, fr = cam.read()
# #     if ret == True:
# #         gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
# #         face_coordinates = classifier.detectMultiScale(gray, 1.3, 5)
# #         for (x, y, w, h) in face_coordinates:
# #             fc = fr[y:y + h, x:x + w, :]
# #             r = cv2.resize(fc, (100, 50)).flatten().reshape(1,-1)
# # # #             # GLCM
# # #             gCoMat = greycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
# # #             contrast = greycoprops(gCoMat, prop='contrast')
# # #             dissimilarity = greycoprops(gCoMat, prop='dissimilarity')
# # #             homogeneity = greycoprops(gCoMat, prop='homogeneity')
# # #             energy = greycoprops(gCoMat, prop='energy')
# # #             correlation = greycoprops(gCoMat, prop='correlation')

# # #             contrast_features = np.reshape(np.array(contrast).ravel(), (1, len(np.array(contrast).ravel())))
# # #             dissimilarity_features = np.reshape(np.array(dissimilarity).ravel(), (1, len(np.array(dissimilarity).ravel())))
# # #             homogeneity_features = np.reshape(np.array(homogeneity).ravel(), (1, len(np.array(homogeneity).ravel())))
# # #             energy_features = np.reshape(np.array(energy).ravel(), (1, len(np.array(energy).ravel())))
# # #             correlation_features = np.reshape(np.array(correlation).ravel(), (1, len(np.array(correlation).ravel())))
            
# # #             features = np.concatenate((contrast_features, dissimilarity_features, homogeneity_features, energy_features,correlation_features), axis=1)
# # #             # if r == 
# # #             text = knn.predict(features)
            
# # #             if text == True:
# # #                 cv2.putText(features, text[0], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
# # #                 cv2.rectangle(features, (x, y), (x + w, y + w), (0, 0, 255), 2)
# # #                 cv2.imshow('livetime face recognition', fr)
# # # #             if cv2.waitKey(1) == ord('q'):
# # # #                     break
# # # #             # else:
# # #             #     cv2.putText(fr, "Unknown", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
# # #             #     cv2.rectangle(fr, (x, y), (x + w, y + w), (0, 0, 255), 2)
# # #             #     cv2.imshow('livetime face recognition', fr)
# #             cv2.imshow('livetime face recognition', fr)
# #             if cv2.waitKey == ord('q'):  
# #                     break
# # #         # else:
# # #             #     print("error")
# # #             #     break



# # # Release the webcam and close the window
# cam.release()
print("execution complete ......THANKU")
cv2.destroyAllWindows()