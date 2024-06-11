import os
import numpy as np
from tqdm import tqdm
import random
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
import gc
from src.image import img, _load_single_image


class dataset:
    def __init__(
        self,
        img_dir,
        standard_size=(32, 32),
        train=True,
        augment_path="",
        label_size_factor=1,
    ):
        time_start = time.time()
        self.standard_size = standard_size
        self.train = train
        self.data = None
        self.images = []
        self.max_label = 0
        self.img_dir = img_dir
        self.load_imgs()
        self.max_label *= label_size_factor
        if augment_path != "" and train:
            self.augment(augment_path, self.max_label)
        self.LabelEncoder = LabelEncoder()
        self.LabelEncoder.fit(
            [
                "none",
                "interdiction",
                "danger",
                "fvert",
                "stop",
                "obligation",
                "forange",
                "ceder",
                "frouge",
            ]
        )
        gc.collect()
        # now preprocess the data
        self.preprocess()

        time_end = time.time()
        print(f"Total loading time: {time_end - time_start:.2f} seconds")

    def preprocess(self):
        chunksize = 100
        chunk_data = []
        imgs_data = []
        self.data = None  # Add this line to initialize self.data
        for img in tqdm(self.images, desc="Preprocessing images"):
            img.preprocess()  # Change this line to directly call preprocess method
            if img.skip:
                print(f"Skipping image {img.name}")
                self.images.remove(img)
                continue
            imgs_data.append(img.data)
        self.data = np.array(imgs_data)
        """
            chunk_data.append(img.data)  # Change img_data to img.data to get the preprocessed data
            if len(chunk_data) == chunksize:
                if self.data is None:
                    self.data = np.array(chunk_data)
                else:
                    self.data = np.concatenate((self.data, np.array(chunk_data)), axis=0)
                chunk_data = []
        if chunk_data:
            if self.data is None:  # Add this check to handle the case when self.data is None
                self.data = np.array(chunk_data)
            else:
                self.data = np.concatenate((self.data, np.array(chunk_data)), axis=0)
        """

    def augment(self, augment_path, target_size):
        augment_label_map = {label: [] for label in self.label_map.keys()}
        img_names = os.listdir(augment_path)

        with ProcessPoolExecutor() as executor:
            results = list(
                executor.map(
                    _load_single_image,
                    [augment_path] * len(img_names),
                    img_names,
                    [self.standard_size] * len(img_names),
                    [self.train] * len(img_names),
                )
            )

        for new_img in results:
            if new_img.skip:
                continue
            else:
                augment_label_map[new_img.label].append(new_img)

        imgs = []
        for label in self.label_map:
            current_size = len(self.label_map[label])
            n_augment = target_size - current_size
            print(f"Augmenting label {label} with {n_augment} images")
            self.augment_label(imgs, augment_label_map[label], n_augment)
        self.images.extend(imgs)
        self.label_map = {}

    def augment_label(self, imgs, augment_imgs, n_augment):
        current_size = len(imgs)
        # suffle the augment_imgs
        random.shuffle(augment_imgs)
        possible_augment = augment_imgs.copy()
        while n_augment > 0:
            self.images.append(possible_augment.pop())
            n_augment -= 1
            if not possible_augment:
                possible_augment = augment_imgs.copy()
                random.shuffle(possible_augment)

    def load_imgs(self):
        self.label_map = {}
        img_names = os.listdir(self.img_dir)
        chunksize = max(1, len(img_names) // (multiprocessing.cpu_count() * 2))
        with ProcessPoolExecutor() as executor:
            results = list(
                executor.map(
                    _load_single_image,
                    [self.img_dir] * len(img_names),
                    img_names,
                    [self.standard_size] * len(img_names),
                    [self.train] * len(img_names),
                    chunksize=chunksize,
                )
            )
        for new_img in results:
            if new_img.skip:
                continue
            self.images.append(new_img)
            if new_img.label not in self.label_map:
                self.label_map[new_img.label] = [new_img]
            else:
                self.label_map[new_img.label].append(new_img)
        max_label = max([len(self.label_map[label]) for label in self.label_map])
        self.max_label = max_label
