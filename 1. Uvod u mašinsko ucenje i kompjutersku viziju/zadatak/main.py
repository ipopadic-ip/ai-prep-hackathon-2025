from process import count_blood_cells
import glob
import sys
import os

DATASET_PATH = '.'+os.path.sep+'dataset'+os.path.sep

processed_image_names = []
red_blood_cell_counts = []

for image_path in glob.glob(DATASET_PATH + "*.bmp"):
    image_directory, image_name = os.path.split(image_path)
    processed_image_names.append(image_name)
    red_blood_cell_counts.append(count_blood_cells(image_path))


result_file_contents = ""
for image_index, image_name in enumerate(processed_image_names):
    result_file_contents += "%s,%s\n" % (image_name, red_blood_cell_counts[image_index])

# čuvanje rezultata u csv fajl
with open('result.csv', 'w') as output_file:
    output_file.write(result_file_contents)
