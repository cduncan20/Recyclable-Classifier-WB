import torchvision
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler, DataLoader
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pathlib
import wandb
from random import random
from random import shuffle
from skimage.util import random_noise


# Obtain list of paths to each image
class ImageFolderWithPaths(torchvision.datasets.ImageFolder):    
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


# Create image with gaussian blur
def gaussian_blur(img):
    image = np.array(img)
    rand_num = random()
    if rand_num <= 0.05:
        image = cv2.GaussianBlur(image, (21, 21), 0)
    return image


# Create image with random noise
def noise(img):
    image = np.array(img)
    rand_num = random()
    if rand_num <= 0.05:
        image = random_noise(image)

        # image is originally a float64 data type, so convert to a uint8 to match others
        image = 255 * image / np.amax(image)  # Current values are between 0 and 1. Change to between 0 and 255
        image = image.astype(np.uint8)  # Convert to uint8 data type
    return image


# Apply transforms to data
def get_transform(transform_dict, train):
    transform_list = [transforms.Resize(256)]
    if train:
        if transform_dict['horizontal']:
            transform_list.append(transforms.RandomHorizontalFlip())
        if transform_dict['vertical']:
            transform_list.append(transforms.RandomVerticalFlip())
        if transform_dict['rot30']:
            transform_list.append(transforms.RandomRotation(degrees=30))
        if transform_dict['noise']:
            transform_list.append(transforms.Lambda(gaussian_blur))
        if transform_dict['blur']:
            transform_list.append(transforms.Lambda(noise))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)


def transform_and_split_data(transform_dict, data_directory_path, val_ratio, test_ratio, batch_size, naming_log):
    # Define transforms for training, validation, and test data
    train_transform = get_transform(transform_dict, train=True)
    val_test_transform = get_transform(transform_dict, train=False)

    # Perform Transformations on Training & Test Data
    # NOTE: Data has not yet been split, so train_data, valid_data & test_data hold reference to all images
    train_set = datasets.ImageFolder(data_directory_path, transform=train_transform)
    val_set = datasets.ImageFolder(data_directory_path, transform=val_test_transform)
    test_set = datasets.ImageFolder(data_directory_path, transform=val_test_transform)

    # Save image Paths
    data_dir = str(data_directory_path)
    dataset = ImageFolderWithPaths(root=data_dir, transform=train_transform) # Obtain list of paths to each image
    data_set_data_dir = DataLoader(dataset=dataset)
    img_paths = []
    for _, data in enumerate(data_set_data_dir):
        _, _, paths = data
        img_paths.append(paths[0])

    # Split data based on test_size ratio defined above
    total_num_images = len(train_set)  # Total number of images
    image_indices = list(range(total_num_images))  # Define an index to each image
    num_val = int(np.floor(val_ratio * total_num_images))  # Number of images assigned to validation data
    num_test = int(np.floor(test_ratio * total_num_images))  # Number of images assigned to testing data

    # Shuffle all image indices so they are in a random order
    c = list(zip(image_indices, img_paths))
    np.random.shuffle(c)  # Shuffle indices and image paths in same shuffled order
    image_indices, img_paths = zip(*c)
    image_indices = list(image_indices)
    img_paths = list(img_paths)

    # Assign image indices belonging to training, validation, & test data
    train_index = image_indices[num_test + num_val:]
    val_index = image_indices[num_test:num_test + num_val]
    test_index = image_indices[:num_test]

    # Assign image paths belonging to training, validation, & test data
    train_paths = img_paths[num_test + num_val:]
    val_paths = img_paths[num_test:num_test + num_val]
    test_paths = img_paths[:num_test]

    # Initialize W&B Run for saving Split Data as W&B Artifact
    run = wandb.init(project=naming_log['project_name'], 
                     name = naming_log['model_name']+"_split", 
                     job_type="data_split")
    run, naming_log = artifact_split_data(run, len(train_paths), train_paths, "Train", naming_log)
    run, naming_log = artifact_split_data(run, len(val_paths), val_paths, "Val", naming_log)
    run, naming_log = artifact_split_data(run, len(test_paths), test_paths, "Test", naming_log)
    run.finish()

    print("Number of training images:", len(train_index))
    print("Number of validation images:", len(val_index))
    print("Number of test images:", len(test_index))

    # Define a random sampler to randomly choose indices for training, validation & test data when loading data
    train_sampler = SubsetRandomSampler(train_index)
    val_sampler = SubsetRandomSampler(val_index)
    test_sampler = SubsetRandomSampler(test_index)

    # Assign split data into loaded data set
    batch_qty = batch_size
    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=batch_qty, num_workers=4)
    val_loader = DataLoader(val_set, sampler=val_sampler, batch_size=batch_qty, num_workers=4)
    test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=batch_qty, num_workers=4)

    return train_loader, val_loader, test_loader, naming_log


# Create List of Classes based on directory subfolders
def get_class_names(data_dir):
    folder_names = []
    for entry_name in os.listdir(data_dir):
        entry_path = os.path.join(data_dir, entry_name)
        if os.path.isdir(entry_path):
            folder_names.append(entry_name)

    folder_names = sorted(folder_names)  # Sort class names alphabetically
    return folder_names

# Save Split Data as W&B Artifact
def artifact_split_data(run, img_qty, img_paths, split_type, naming_log):
    artifact_name = "_".join(["artifact", split_type, "data"])
    naming_log[artifact_name] = "_".join([split_type, "Data", str(img_qty)])

    # create an artifact for all the raw data
    split_data_at = wandb.Artifact(naming_log[artifact_name], type="split_data")
    for file in img_paths:
        split_file = file.split("/")
        split_file = split_file[-2:]
        split_data_at.add_file(file, name=split_file[0] + "/" + split_file[1])
    
    # save artifact to W&B
    run.use_artifact(naming_log['project_name'] + '/' + naming_log['artifact_raw_data'] + ":latest")
    run.log_artifact(split_data_at)

    return run, naming_log

# Save Raw Data as W&B Artifact
def artifact_raw_data(data_directory_path, naming_log):
    # Initialize W&B Run
    run = wandb.init(project=naming_log['project_name'], 
                     name = naming_log['model_name']+"_raw_data", 
                     job_type="upload")

    # create an artifact for all the raw data
    data = datasets.ImageFolder(data_directory_path)
    total_images = len(data)
    naming_log['artifact_raw_data'] = "Raw_Data_" + str(total_images)
    raw_data_artifact = wandb.Artifact(naming_log['artifact_raw_data'], type="raw_data")

    # Obtain file path of each individual photo
    labels = os.listdir(data_directory_path)
    for l in labels:
        imgs_per_label = os.path.join(data_directory_path, l)
        if os.path.isdir(imgs_per_label):
            imgs = os.listdir(imgs_per_label)
            _, _, files = next(os.walk(imgs_per_label))
            IMAGES_PER_LABEL = len(files)
            # randomize the order
            shuffle(imgs)
            img_file_ids = imgs[:IMAGES_PER_LABEL]
            for f in img_file_ids:
                file_path = os.path.join(data_directory_path, l, f)
                # add file to artifact by full path
                raw_data_artifact.add_file(file_path, name=l + "/" + f)

    # save artifact to W&B
    run.log_artifact(raw_data_artifact)
    run.finish()
    return naming_log


def load_data(hyperparameters, naming_log):
    cwd = pathlib.Path.cwd()
    data_dir = cwd.joinpath("csci508_final", "Images", "TRAINING_&_TEST", "TRAINING_&_TEST_IMAGES")  # Image file path
    class_names = get_class_names(data_dir)

    # Save Raw Data as W&B Artifact
    naming_log = artifact_raw_data(data_dir, naming_log)

    # Create dictionary of augmentations
    aug_dict = dict([('horizontal', hyperparameters['transform_horz']),
                     ('vertical', hyperparameters['transform_vert']),
                     ('rot30', hyperparameters['transform_rot']),
                     ('noise', hyperparameters['transform_noise']),
                     ('blur', hyperparameters['transform_blur'])])

    # Create Data Loaders by splitting data & applying transforms
    train_loader, val_loader, test_loader, naming_log = transform_and_split_data(aug_dict, 
                                                                                 data_dir, 
                                                                                 hyperparameters['val_ratio'], 
                                                                                 hyperparameters['test_ratio'], 
                                                                                 hyperparameters['batch_size'],
                                                                                 naming_log)

    return train_loader, val_loader, test_loader, class_names, naming_log, aug_dict


if __name__ == '__main__':
    sys.exit(load_data())
