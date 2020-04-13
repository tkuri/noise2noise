from pathlib import Path
import random
import numpy as np
import cv2
from keras.utils import Sequence
import os


class NoisyImageGenerator(Sequence):
    def __init__(self, image_dir, source_noise_model, target_noise_model, batch_size=32, image_size=64):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
        self.image_paths = [p for p in Path(image_dir).glob("**/*") if p.suffix.lower() in image_suffixes]
        self.source_noise_model = source_noise_model
        self.target_noise_model = target_noise_model
        self.image_num = len(self.image_paths)
        self.batch_size = batch_size
        self.image_size = image_size

        if self.image_num == 0:
            raise ValueError("image dir '{}' does not include any image".format(image_dir))

    def __len__(self):
        return self.image_num // self.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        y = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        sample_id = 0

        while True:
            image_path = random.choice(self.image_paths)
            image = cv2.imread(str(image_path))
            h, w, _ = image.shape

            if h >= image_size and w >= image_size:
                h, w, _ = image.shape
                i = np.random.randint(h - image_size + 1)
                j = np.random.randint(w - image_size + 1)
                clean_patch = image[i:i + image_size, j:j + image_size]
                x[sample_id] = self.source_noise_model(clean_patch)
                y[sample_id] = self.target_noise_model(clean_patch)

                sample_id += 1

                if sample_id == batch_size:
                    return x, y

class NoisyImageGeneratorIn2(Sequence):
    def __init__(self, image_dir_A, image_dir_B, source_noise_model, target_noise_model, batch_size=32, image_size=64):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
        self.image_paths_A = [p for p in Path(image_dir_A).glob("**/*") if p.suffix.lower() in image_suffixes]
        self.image_paths_B = [p for p in Path(image_dir_B).glob("**/*") if p.suffix.lower() in image_suffixes]
        self.source_noise_model = source_noise_model
        self.target_noise_model = target_noise_model
        self.image_num_A = len(self.image_paths_A)
        self.image_num_B = len(self.image_paths_B)
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_dir_B = image_dir_B

        if self.image_num_A == 0 or self.image_num_B == 0:
            raise ValueError("image dir '{}' does not include any image".format(image_dir))

        if self.image_num_A != self.image_num_B:
            raise ValueError("image_num_A '{}' does not match image_num_B '{}'".format(image_num_A, image_num_B))
            
    def __len__(self):
        return self.image_num_A // self.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        y = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        sample_id = 0

        while True:
            image_path_A = random.choice(self.image_paths_A)
            image_A = cv2.imread(str(image_path_A))
            image_B = cv2.imread(self.image_dir_B + '/' + os.path.basename(image_path_A))
#             print('image_A:', str(image_path_A))
#             print('image_B:', str(self.image_dir_B + '/' + os.path.basename(image_path_A)))
#             print('image_A.shape ', image_A.shape)
#             print('image_B.shape ', image_B.shape)
            if image_A.shape != image_B.shape:
                raise ValueError("image_A.shape '{}' does not match image_B.shape '{}'".format(image_A.shape, image_B.shape))
            h, w, _ = image_A.shape

            if h >= image_size and w >= image_size:
                h, w, _ = image_A.shape
                i = np.random.randint(h - image_size + 1)
                j = np.random.randint(w - image_size + 1)
                patch_A = image_A[i:i + image_size, j:j + image_size]
                patch_B = image_B[i:i + image_size, j:j + image_size]
                x[sample_id] = self.source_noise_model(patch_A)
                y[sample_id] = self.target_noise_model(patch_B)

                sample_id += 1

                if sample_id == batch_size:
                    return x, y

                
                
class ValGenerator(Sequence):
    def __init__(self, image_dir, val_noise_model):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
        image_paths = [p for p in Path(image_dir).glob("**/*") if p.suffix.lower() in image_suffixes]
        self.image_num = len(image_paths)
        self.data = []

        if self.image_num == 0:
            raise ValueError("image dir '{}' does not include any image".format(image_dir))

        for image_path in image_paths:
            y = cv2.imread(str(image_path))
            h, w, _ = y.shape
            y = y[:(h // 16) * 16, :(w // 16) * 16]  # for stride (maximum 16)
            x = val_noise_model(y)
            self.data.append([np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)])

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        return self.data[idx]

    
class ValGeneratorIn2(Sequence):
    def __init__(self, image_dir_A, image_dir_B, val_noise_model):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
        image_paths_A = [p for p in Path(image_dir_A).glob("**/*") if p.suffix.lower() in image_suffixes]
        image_paths_B = [p for p in Path(image_dir_B).glob("**/*") if p.suffix.lower() in image_suffixes]
        self.image_num_A = len(image_paths_A)
        self.image_num_B = len(image_paths_B)
        self.data = []

        if self.image_num_A == 0:
            raise ValueError("image dir '{}' does not include any image".format(image_dir_A))

        if self.image_num_A != self.image_num_B:
            raise ValueError("image_num_A '{}' does not match image_num_B '{}'".format(image_num_A, image_num_B))
            
        for image_path_A, image_path_B in zip(image_paths_A, image_paths_B):
            x = cv2.imread(str(image_path_A))
            y = cv2.imread(str(image_path_B))
            h, w, _ = y.shape
            x = y[:(h // 16) * 16, :(w // 16) * 16]  # for stride (maximum 16)
            y = y[:(h // 16) * 16, :(w // 16) * 16]  # for stride (maximum 16)
#             x = val_noise_model(y)
            self.data.append([np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)])

    def __len__(self):
        return self.image_num_A

    def __getitem__(self, idx):
        return self.data[idx]