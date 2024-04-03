#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from PIL import Image

# Specify the root directory where your dataset is stored
root_directory = "C:\\Users\\Farah\\Documents\\4.YIL\\BİLGİSAYAR GÖRMESİNE GİRİŞ\\odev1\\OvarianCancer"

# Get the list of class folders
class_folders = ["Clear_Cell", "Endometri", "Mucinous", "Non_Cancerous", "Serous"]

# Görüntüleri ve etiketleri ayıralım
images = []
labels = []

allowed_extensions = (".jpg", ".JPG")  # Add more extensions as needed

for class_label, class_folder in enumerate(class_folders):
    class_folder_path = os.path.join(root_directory, class_folder)

    for filename in os.listdir(class_folder_path):
        if filename.lower().endswith(allowed_extensions):
            image_path = os.path.join(class_folder_path, filename)
            print(f"Reading image from: {image_path}")

            try:
                # Read the image using PIL
                image = Image.open(image_path)
                image = np.array(image)

                # Convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Append to lists
                images.append(image)
                labels.append(class_label)
            except Exception as e:
                print(f"Error: Unable to read image at {image_path}. {e}")

# Print some debug information
print(f"Number of images: {len(images)}")
print(f"Number of labels: {len(labels)}")


# In[5]:


import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ... (Previous code)

# Hierarchical image processing function
def hierarchical_image_processing(image):
    processed_images = []

    # Level 1: Basic preprocessing
    resized_image = cv2.resize(image, (300, 300))
    lab_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2Lab)
    l_channel, a, b = cv2.split(lab_image)
    processed_images.append(l_channel)

    # Level 2: Additional color space transformation
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2HSV)
    h_channel, s, v = cv2.split(hsv_image)
    processed_images.append(h_channel)

    # Level 3: Edge detection
    edges_image = cv2.Canny(l_channel, 50, 150)
    processed_images.append(edges_image)

    # Level 4: Custom processing (example: gamma correction)
    gamma_value = 1.5
    gamma_corrected_image = np.power(l_channel / float(np.max(l_channel)), gamma_value)
    processed_images.append(gamma_corrected_image)

    return processed_images

# Apply hierarchical image processing to all images
for image_index, image in enumerate(images):
    print(f"Processing image {image_index + 1}/{len(images)}")

    # Step 1: Apply hierarchical image processing
    processed_images = hierarchical_image_processing(image)

    # Step 2: Save processed images
    output_directory = "processed_images"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for i, processed_image in enumerate(processed_images):
        output_path = os.path.join(output_directory, f"processed_image_{image_index}_level_{i + 1}.jpg")
        cv2.imwrite(output_path, processed_image)
        print(f"Processed image saved to: {output_path}")


# In[4]:





# In[ ]:




