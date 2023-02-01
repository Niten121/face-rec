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


dataset = pd.read_csv('features4.csv')
print(dataset)
print("shape", dataset.shape)


print(dataset.info())
print(dataset.isnull().sum())

# # Encode labels
le = LabelEncoder()
lable = le.fit_transform(dataset.Outcome)
print(lable)

dataset["outcome"]=lable
print(dataset.head())

# x=dataset.drop(['name'],axis='columns')
# y=dataset.name

# print(x)
# print(y)


# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

# # Standardize features by removing the mean and scaling to unit variance.
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)


# print('x.shape,x_test.shape,x_train.shape')
# print(x.shape,x_test.shape,x_train.shape)
# knn = KNeighborsClassifier(n_neighbors = 1)
# knn.fit(x_train, y_train)
# print('fitting done')
#  # Make predictions on the test data
# y_pred = knn.predict(x_test)
# print("prediction done")
# print(y_pred)
# print(x_test.shape)
# y_pred1=knn.predict([[8.64712406015038,1.15545739348371,0.679766942410002,0.0657284068833956,0.997086148615946]])
# y_pred2 = knn.predict([[2.70988095238095,0.794968671679198,0.70893501736738,0.0658268320950465,0.998957953762544]])
# print(y_pred1)
# print(y_pred2)

# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

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
                
#                 if y_pred == 0:
#                     cv2.putText("niten", y_pred[0].all, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
#                     cv2.rectangle(features, (x, y), (x + w, y + w), (0, 0, 255), 2)
#                     cv2.imshow('livetime face recognition', fr)
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
cv2.destroyAllWindows()