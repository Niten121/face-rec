# import cv2
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import LabelEncoder

# # Collect facial data and labels
# # and store them in a CSV file
# # ...

# # Read data from CSV file
# data = np.genfromtxt('features4.csv', delimiter=',')
# X = data[:, :-1]
# y = data[:, -1]

# # Encode labels
# le = LabelEncoder()
# y = le.fit_transform(y)

# # Train KNN model
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X, y)

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

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Read data from CSV file
data = pd.read_csv('features4.csv')
X=data.drop(['name'],axis='columns')
y=data.name

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

with open('features4.csv', "w", newline="") as wr:
    writer = csv.writer(wr)
    writer.writerow(["contrast_features", "dissimilarity_features", "homogeneity_features", "energy_features","correlation_features","name","outcome"])

     for f in files:
     label = f.split('\\')[-1]
     img = cv2.imread(f)
     img1 = cv2.resize(img, (400, 400))
     c = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
     # conversion of RGB images into Gray Scale
     gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)

     # LBP
     # feat_lbp = local_binary_pattern(gray, 5, 2, 'uniform')
     # lbp_hist, _ = np.histogram(feat_lbp, 8)
     # lbp_hist = np.array(lbp_hist, dtype=float)
     # lbp_prob = np.divide(lbp_hist, np.sum(lbp_hist))
     # lbp_energy = np.nansum(lbp_prob ** 2)
     # lbp_entropy = -np.nansum(np.multiply(lbp_prob, np.log2(lbp_prob)))

     # lbphist_features = np.reshape(np.array(lbp_hist).ravel(), (1, len(np.array(lbp_hist).ravel())))
     # lbpprob_features = np.reshape(np.array(lbp_prob).ravel(), (1, len(np.array(lbp_prob).ravel())))
     # lbpenrgy_features = np.reshape(np.array(lbp_energy).ravel(), (1, len(np.array(lbp_energy).ravel())))
     # lbpento_features = np.reshape(np.array(lbp_entropy).ravel(), (1, len(np.array(lbp_entropy).ravel())))

     # GLCM
     gCoMat = greycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
     contrast = greycoprops(gCoMat, prop='contrast')
     dissimilarity = greycoprops(gCoMat, prop='dissimilarity')
     homogeneity = greycoprops(gCoMat, prop='homogeneity')
     energy = greycoprops(gCoMat, prop='energy')
     correlation = greycoprops(gCoMat, prop='correlation')

     contrast_features = np.reshape(np.array(contrast).ravel(), (1, len(np.array(contrast).ravel())))
     dissimilarity_features = np.reshape(np.array(dissimilarity).ravel(), (1, len(np.array(dissimilarity).ravel())))
     homogeneity_features = np.reshape(np.array(homogeneity).ravel(), (1, len(np.array(homogeneity).ravel())))
     energy_features = np.reshape(np.array(energy).ravel(), (1, len(np.array(energy).ravel())))
     correlation_features = np.reshape(np.array(correlation).ravel(), (1, len(np.array(correlation).ravel())))
     
     features = np.concatenate((contrast_features, dissimilarity_features, homogeneity_features, energy_features,correlation_features), axis=1);
     ff = features[0].tolist()
     writer.writerow(ff + [labels[i]])
     i += 1
 wr.close()











# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in frame
    face_cascade = cv2.CascadeClassifier('C:/Users/sethy/PycharmProjects/faceRecognition/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    # faces = detect_faces(frame)

    for(x,y,w,h) in faces:
        gCoMat = greycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
        contrast = greycoprops(gCoMat, prop='contrast')
        dissimilarity = greycoprops(gCoMat, prop='dissimilarity')
        homogeneity = greycoprops(gCoMat, prop='homogeneity')
        energy = greycoprops(gCoMat, prop='energy')
        correlation = greycoprops(gCoMat, prop='correlation')

        contrast_features = np.reshape(np.array(contrast).ravel(), (1, len(np.array(contrast).ravel())))
        dissimilarity_features = np.reshape(np.array(dissimilarity).ravel(), (1, len(np.array(dissimilarity).ravel())))
        homogeneity_features = np.reshape(np.array(homogeneity).ravel(), (1, len(np.array(homogeneity).ravel())))
        energy_features = np.reshape(np.array(energy).ravel(), (1, len(np.array(energy).ravel())))
        correlation_features = np.reshape(np.array(correlation).ravel(), (1, len(np.array(correlation).ravel())))
        
        features = np.concatenate((contrast_features, dissimilarity_features, homogeneity_features, energy_features,correlation_features), axis=1)

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = knn.predict(X,y)

        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            id = 1
            confidence = " niten {0}%".format(round( confidence))
        elif (confidence < 100):
            id = 2
            confidence = " raghav {0}%".format(round( confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(frame, str(id), (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, str(confidence), (x+5,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',frame) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cap.release()
cv2.destroyAllWindows()