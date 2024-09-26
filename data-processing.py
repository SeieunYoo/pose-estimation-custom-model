import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO

# Path to COCO annotation file
ann_file = 'annotations/person_keypoints_train2017.json'

# Load annotation file using COCO API
coco = COCO(ann_file)

# Path to COCO image folder
img_folder = 'train2017'

# Get image IDs for the person category
cat_ids = coco.getCatIds(catNms=['person'])
img_ids = coco.getImgIds(catIds=cat_ids)

# Define custom Dataset class
class COCODataset(Dataset):
    def __init__(self, img_folder, coco, img_ids):
        self.img_folder = img_folder
        self.coco = coco
        self.img_ids = img_ids

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_data = self.coco.loadImgs(img_id)[0]

        # Set image path
        img_path = os.path.join(self.img_folder, img_data['file_name'])

        # Load and preprocess image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))  # Resize to model input size
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0  # Convert to Tensor (FloatTensor)

        # Load annotations (keypoints)
        ann_ids = self.coco.getAnnIds(imgIds=img_data['id'], catIds=self.coco.getCatIds(catNms=['person']), iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        keypoints = np.zeros((17, 3))  # 17 keypoints
        for ann in anns:
            kp = np.array(ann['keypoints']).reshape(-1, 3)
            keypoints[:kp.shape[0], :] = kp

        # Convert keypoints to float32 type
        return image, torch.tensor(keypoints, dtype=torch.float32)

# Create COCO Dataset object
coco_dataset = COCODataset(img_folder=img_folder, coco=coco, img_ids=img_ids)
