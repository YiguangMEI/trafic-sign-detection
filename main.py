from src.dataset import dataset
from src.train import train
from src.augment import ImageBatchProcessor
from src.crop import crop_images_from_folder

# from src.CNNmodel import CNNmodel

if __name__ == "__main__":
    # data_augment()

    image_folder = "dataset2/train/images/"
    label_folder = "dataset2/train/labels/"
    target_image_folder = "dataset2/train/images_aug/"
    target_label_folder = "dataset2/train/labels_aug/"

    """
    processor = ImageBatchProcessor(image_folder, label_folder, target_image_folder, target_label_folder)
    processor.augment_and_save(nb_augmentation=5)
    crop_images_from_folder(target_image_folder, target_label_folder, 'dataset2/train/aug_cropped_images/')
    val_path = 'dataset2/val/images/'
    val_label_path = 'dataset2/val/labels/'
    crop_images_from_folder(val_path, val_label_path, 'dataset2/val/cropped_images/')
    """
    train_path = "dataset2/train/cropped_images/"
    val_path = "dataset2/val/cropped_images/"
    aug_train_path = "dataset2/train/aug_cropped_images/"
    # train(train_path=train_path, aug_train_path=aug_train_path, val_path=val_path, seed=42)
    train(
        train_path=train_path, aug_train_path=aug_train_path, val_path=val_path, seed=42
    )

    """
    train_path = 'data/train/cropped_images/'
    train_data = dataset(train_path, standard_size=(32, 32))
    val_path = 'data/val/cropped_images/'
    val_data = dataset(val_path, standard_size=(32, 32))
    cnn = CNNmodel((32, 32, 3), 8)

    cnn.train(train_data, batch_size=32, epochs=100)
    accuracy = cnn.evaluate(val_data)
    print("Model accuracy with CNN: {}".format(accuracy))
"""
