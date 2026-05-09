import os
import cv2
import numpy as np
import scipy.io as sio

NORMALIZED_MAT_ROOT = "data/original/MPIIGaze/Data/Normalized"
# Create a new folder to store the extracted images
EXTRACTED_IMG_ROOT = "data/MPIIGaze_Normalized_Images"

def extract_normalized_images():
    if not os.path.exists(EXTRACTED_IMG_ROOT):
        os.makedirs(EXTRACTED_IMG_ROOT)

    # Loop through all subject folders (p00 to p14)
    for subject in os.listdir(NORMALIZED_MAT_ROOT):
        subject_path = os.path.join(NORMALIZED_MAT_ROOT, subject)
        if not os.path.isdir(subject_path):
            continue
            
        out_subject_path = os.path.join(EXTRACTED_IMG_ROOT, subject)
        os.makedirs(out_subject_path, exist_ok=True)

        # Loop through all days for the subject
        for mat_file in os.listdir(subject_path):
            if not mat_file.endswith('.mat'):
                continue
                
            day = mat_file.split('.')[0]
            mat_path = os.path.join(subject_path, mat_file)
            mat = sio.loadmat(mat_path)
            
            out_day_path = os.path.join(out_subject_path, day)
            os.makedirs(out_day_path, exist_ok=True)
            
            # Extract both left and right eye images
            for eye in ['left', 'right']:
                try:
                    images = mat['data'][0][0][eye][0][0]['image']
                    
                    # Save each frame as a .jpg
                    for frame_index in range(images.shape[0]):
                        img_array = images[frame_index]
                        
                        # Format: p00/day01/left_0001.jpg
                        # (Padding with zeros to match your extract_frame_index logic)
                        filename = f"{eye}_{frame_index + 1:04d}.jpg"
                        save_path = os.path.join(out_day_path, filename)
                        
                        cv2.imwrite(save_path, img_array)
                        
                except KeyError:
                    # Sometimes a day might be missing data for one eye
                    continue
                    
        print(f"Finished extracting images for {subject}")

if __name__ == "__main__":
    extract_normalized_images()