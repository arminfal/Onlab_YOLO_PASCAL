import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from scipy import stats
import glob
import os
import skimage.measure
import cv2
import shutil
def is_clockwise(contour):
    # calculate the signed area
    signed_area = 0.5 * np.sum((contour[:-1, 0] * contour[1:, 1]) - (contour[:-1, 1] * contour[1:, 0]))
    # If the signed area is negative, the points are in clockwise order
    return signed_area < 0
# Classes
class_names = {
    "sofa": 0,
    "train": 1,
    "bicycle": 2,
    "horse": 3,
    "motorbike": 4,
    "sheep": 5,
    "cat": 6,
    "bird": 7,
    "bottle": 8,
    "cow": 9,
    "bus": 10,
    "pottedplant": 11,
    "diningtable": 12,
    "person": 13,
    "tvmonitor": 14,
    "chair": 15,
    "dog": 16,
    "boat": 17,
    "car": 18,
    "aeroplane": 19
}


def generate_labels(xml_file, png_file, label_file):
    # Parse XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Load PNG mask in P format
    img = Image.open(png_file).convert('P')
    mask = np.array(img)

    # Get image dimensions for normalization
    width, height = img.size

    # Initialize color-object mapping and class name mapping
    color_object_mapping = {}
    class_name_mapping = {}

    # Get all objects
    objects = list(root.findall('object'))

    # For each object in the XML file
    for idx, obj in enumerate(objects):
        # Get class name and bounding box
        class_name = obj.find('name').text
        class_name_mapping[idx] = class_name 
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # Extract region from bounding box coordinates
        region = mask[ymin:ymax, xmin:xmax]

        # Exclude edge color(255) and black(0)
        region = region[(region != 0) & (region != 255)]

        # For each unique color in the region
        for color_index in np.unique(region):
            # Map color index to list of objects
            color_object_mapping.setdefault(color_index, []).append(idx)

     # Open label file for writing
    with open(label_file, 'w') as f:
        # For each unique color index in mask
        for color_index in np.unique(mask.reshape(-1)):
            # If color index is mapped to an object
            if color_index in color_object_mapping:
                # Get object IDs
                obj_ids = color_object_mapping[color_index]

                # Find contours of this color index and add a border to avoid edge cases
                border_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
                contours = skimage.measure.find_contours(border_mask == color_index, 0.5)

                for contour in contours:
                    # Ensure contour is in clockwise order
                    if not is_clockwise(contour):
                        contour = contour[::-1]

                    # Check if contour size is less than 10
                    if contour.shape[0] < 10:
                        continue

                    # Calculate bounding box of contour, so it can be compared later to the bounding box from the xml files
                    min_y, min_x = np.min(contour, axis=0)
                    max_y, max_x = np.max(contour, axis=0)

                    # Initialize max intersection to a default value,it is used to find the most similar bounding box
                    max_intersection = 0

                    # Find object whose bounding box best matches the contour's bounding box
                    for obj_id in obj_ids:
                        obj = objects[obj_id]
                        bndbox = obj.find('bndbox')
                        xmin = int(bndbox.find('xmin').text)
                        ymin = int(bndbox.find('ymin').text)
                        xmax = int(bndbox.find('xmax').text)
                        ymax = int(bndbox.find('ymax').text)

                        # Calculate intersection of object's bounding box with contour's bounding box
                        intersection = max(0, min(xmax, max_x) - max(xmin, min_x)) * max(0, min(ymax, max_y) - max(ymin, min_y))

                        # Calculate area of contour's bounding box
                        contour_area = (max_x - min_x) * (max_y - min_y)

                        class_name = class_name_mapping[obj_id]  # assign class number, so if the if part doesn't run, there will still be an assigned class number
                        class_number = class_names[class_name]
                        # Initialize max_ratio, max_area and class_number to default values
                        max_ratio = 0
                        max_area = float('inf')

                        # Find object whose bounding box best matches the contour's bounding box
                        for obj_id in obj_ids:
                            obj = objects[obj_id]
                            bndbox = obj.find('bndbox')
                            xmin = int(bndbox.find('xmin').text)
                            ymin = int(bndbox.find('ymin').text)
                            xmax = int(bndbox.find('xmax').text)
                            ymax = int(bndbox.find('ymax').text)

                            # Calculate intersection of object's bounding box with contour's bounding box
                            intersection = max(0, min(xmax, max_x) - max(xmin, min_x)) * max(0, min(ymax, max_y) - max(ymin, min_y))

                            # Calculate area of contour's bounding box
                            contour_area = (max_x - min_x) * (max_y - min_y)

                            # Calculate area of object's bounding box
                            object_area = (xmax - xmin) * (ymax - ymin)

                            # Calculate ratio of intersection area to contour area
                            ratio = intersection / contour_area if contour_area > 0 else 0

                            # If ratio is larger than the current maximum ratio, or if ratio is equal to the maximum ratio and object area is smaller than the current maximum area, then update the maximum ratio, maximum area, and class name+number
                            if ratio > max_ratio or (ratio == max_ratio and object_area < max_area):
                                max_ratio = ratio
                                max_area = object_area
                                class_name = class_name_mapping[obj_id]  # Get class name from dictionary
                                class_number = class_names[class_name]

                    # Initialize a string to store all contours of an object
                    contour_str = ''
                    for y, x in contour:
                        # Normalize coordinates and remove the added border
                        contour_str += f' {(x-1)/width} {(y-1)/height}'

                    # Write class number and all contour coordinates to label file
                    f.write(f'{class_number}{contour_str}\n')

# Get all XML files in folder
xml_files = glob.glob('Annotations\\*.xml')
png_files_dir = 'SegmentationObject\\'
label_files_dir = 'IMGAnnotations\\'

# For each XML file
for xml_file in xml_files:
    # Extract the base name
    base_name = os.path.splitext(os.path.basename(xml_file))[0]
    
    # Generate PNG and TXT file paths
    png_file = os.path.join(png_files_dir, base_name + '.png')
    label_file = os.path.join(label_files_dir, base_name + '.txt')

    # Check if PNG file exists
    if not os.path.exists(png_file):
        continue

    # Generate labels
    generate_labels(xml_file, png_file, label_file)



# Define the directories
base_dir_images = 'JPEGImages'
base_dir_labels = 'IMGAnnotations'
train_dir_images = 'dataset/train/images'
train_dir_labels = 'dataset/train/labels'
valid_dir_images = 'dataset/valid/images'
valid_dir_labels = 'dataset/valid/labels'
test_dir_images = 'dataset/test/images'
test_dir_labels = 'dataset/test/labels'

# Create the directories
os.makedirs(train_dir_images, exist_ok=True)
os.makedirs(train_dir_labels, exist_ok=True)
os.makedirs(valid_dir_images, exist_ok=True)
os.makedirs(valid_dir_labels, exist_ok=True)
os.makedirs(test_dir_images, exist_ok=True)
os.makedirs(test_dir_labels, exist_ok=True)

# Get all the labels in the base directory
# Checking labels, because the images not necessarily have labels
labels = [f for f in os.listdir(base_dir_labels) if f.endswith('.txt')]

# Sort the labels
labels.sort()

# Create a list of corresponding images
images = [f.replace('.txt', '.jpg') for f in labels]

# Shuffle the indices
indices = np.arange(len(labels))
np.random.shuffle(indices)

# Split the indices into 70% train, 15% validation, and 15% test
train_indices, validate_indices, test_indices = np.split(indices, [int(.7*len(indices)), int(.85*len(indices))])

# Copy the images and labels into the right directories
for idx in train_indices:
    shutil.copy(os.path.join(base_dir_images, images[idx]), os.path.join(train_dir_images, images[idx]))
    shutil.copy(os.path.join(base_dir_labels, labels[idx]), os.path.join(train_dir_labels, labels[idx]))
for idx in validate_indices:
    shutil.copy(os.path.join(base_dir_images, images[idx]), os.path.join(valid_dir_images, images[idx]))
    shutil.copy(os.path.join(base_dir_labels, labels[idx]), os.path.join(valid_dir_labels, labels[idx]))
for idx in test_indices:
    shutil.copy(os.path.join(base_dir_images, images[idx]), os.path.join(test_dir_images, images[idx]))
    shutil.copy(os.path.join(base_dir_labels, labels[idx]), os.path.join(test_dir_labels, labels[idx]))