import os
import cv2
import numpy as np


# Define the path of the parent directory
parent_dir = "collected_data"
# Define the path of the output folder
output_dir = "processed_data"

print("process stared kindly wait ............ ")

# Create the output folder if it doesn't exist
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Loop through all the subfolders of the parent directory
for subdir_name in os.listdir(parent_dir):
    # Create the path to the subfolder
    subdir_path = os.path.join(parent_dir, subdir_name)

    # Check if the subfolder is a directory
    if os.path.isdir(subdir_path):
        # Create the output subfolder
        output_subdir_path = os.path.join(output_dir, subdir_name)
        if not os.path.exists(output_subdir_path):
            os.mkdir(output_subdir_path)
        # Loop through all the files in the subfolder
        for filename in os.listdir(subdir_path):
            # Create the path to the file
            file_path = os.path.join(subdir_path, filename)

            # Check if the file is an image
            if file_path.endswith(".jpg") or file_path.endswith(".png"):
                # Load the image
                gray = cv2.imread(file_path)
                cv2.imshow("original",gray)
            
            
                # Remove noise using a Gaussian blur
                blurred = cv2.GaussianBlur(gray, (3, 3), 0)
                cv2.imshow("blured",blurred)

                # Sharpen the image using a unsharp mask
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                sharpened = cv2.filter2D(blurred, -1, kernel)
                cv2.imshow("sharpened",sharpened)
                
        
                # Normalize the image
                normalized = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)

                cv2.imshow("normalized",normalized)
                
                # denoising
                noise_1 = cv2.fastNlMeansDenoising(normalized, 2, 3.0, 7, 21)
                cv2.imshow('denoise', noise_1)
                
              
                processed_image = noise_1
                # Save the processed image
                cv2.imwrite(os.path.join(output_subdir_path, filename), processed_image)
                cv2.waitKey(1)

cv2.waitKey(0)
print("pre-processing done .... ")
print("MOVE FORWARD--------------->>>>>>>Feature Extraction")
cv2.destroyAllWindows()
