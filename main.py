#!/usr/bin/env python3.6

import argparse
import sys
import getopt
import os
import pathlib

import csci508_final as csci


def main(argv):
    # Get full command-line arguments
    full_cmd_arguments = sys.argv

    # Keep all but the first
    argument_list = full_cmd_arguments[1:]

    short_options = "h"
    long_options = ["help",
                    "train_ratio=",
                    "batch_size=",
                    "learn_rate=",
                    "epoch_qty=",
                    "transform_horz=",
                    "transform_vert=",
                    "transform_rot30=",
                    "transform_noise=",
                    "transform_blur=",
                    "architecture="]

    try:
        arguments, values = getopt.getopt(argument_list, short_options, long_options)
    except getopt.error as err:
        # Output error, and return with an error code
        print (str(err))
        sys.exit(2)

    for current_argument, current_value in arguments:
        if current_argument in ("-h", "--help"):
            print()
            print("Usage: main.py [argument]")
            print()
            print("Optional Arguments:")
            print("-h, --help \t\t\t Show this help message and exit")
            print("--train_ratio=[float] \t\t Training ratio for splitting data. Chosen value should be a float < 1.00.")
            print("\t\t\t\t Default value is 0.6.")
            print("--batch_size=[int] \t\t Batch size for training data. Chosen value should be an integer < 50.")
            print("\t\t\t\t Default value is 8.")
            print("--learn_rate=[float] \t\t Learning rate for training model. Chosen value should be a float.")
            print("\t\t\t\t Default value is 0.001.")
            print("--epoch_qty=[int] \t\t Epoch Quantity for training model. Chosen value should be an integer.")
            print("\t\t\t\t Default value is 8.")
            print("--transform_horz=[bool] \t Horizontal Augmentation for training data. Chosen value should be True or False.")
            print("\t\t\t\t Default value is False.")
            print("--transform_vert=[bool] \t Vertical Augmentation for training data. Chosen value should be True or False.")
            print("\t\t\t\t Default value is False.")
            print("--transform_rot30=[bool] \t 30 Degree Rotation Augmentation for training data. Chosen value should be True or False.")
            print("\t\t\t\t Default value is False.")
            print("--transform_noise=[bool] \t Noise Augmentation for training data. Chosen value should be True or False.")
            print("\t\t\t\t Default value is False.")
            print("--transform_blur=[bool] \t Blur Augmentation for training data. Chosen value should be True or False.")
            print("\t\t\t\t Default value is False.")
            print("--architecture=[string] \t Pre-Trained model type for leveraging transfer learning. Chosen value should be")
            print("\t\t\t\t 'resnet18', 'resnet34', 'resnet50', or 'resnet50-modnet'. Default value is 'resnet18'.")
            print()
            exit()
        elif current_argument in ("--train_ratio"):
            train_ratio = float(current_value)
        elif current_argument in ("--epoch_qty"):
            epoch_qty = int(current_value)
        elif current_argument in ("--batch_size"):
            batch_size = int(current_value)
        elif current_argument in ("--learn_rate"):
            learn_rate = float(current_value)
        elif current_argument in ("--transform_horz"):
            transform_horz = bool(current_value)
        elif current_argument in ("--transform_vert"):
            transform_horz = bool(current_value)
        elif current_argument in ("--transform_rot30"):
            transform_horz = bool(current_value)
        elif current_argument in ("--transform_noise"):
            transform_horz = bool(current_value)
        elif current_argument in ("--transform_blur"):
            transform_horz = bool(current_value)
        elif current_argument in ("--architecture"):
            model_type = current_value


    if 'train_ratio' not in locals():
        train_ratio = 0.6
    if 'epoch_qty' not in locals():
        epoch_qty = 5
    if 'batch_size' not in locals():
        batch_size = 8
    if 'learn_rate' not in locals():
        learn_rate = 1e-3
    if 'transform_horz' not in locals():
        transform_horz = False
    if 'transform_vert' not in locals():
        transform_vert = False
    if 'transform_rot' not in locals():
        transform_rot = False
    if 'transform_noise' not in locals():
        transform_noise = False
    if 'transform_blur' not in locals():
        transform_blur = False
    if 'model_type' not in locals():
        model_type = 'resnet18'
    
    val_ratio = (1.0 - train_ratio)/2
    test_ratio = 1.0 - train_ratio - val_ratio
    
    hyperparameters = dict(
        train_ratio = train_ratio,
        val_ratio = val_ratio,
        test_ratio = test_ratio,
        epoch_qty = epoch_qty,
        batch_size = batch_size,
        learn_rate = learn_rate,
        transform_horz = transform_horz,
        transform_vert = transform_vert,
        transform_rot = transform_rot,
        transform_noise = transform_noise,
        transform_blur = transform_blur,
        architecture = model_type
    )

    model_dict, model_file_name = csci.model_selection.model_selection(model_type)
    csci.train_test_model.main(hyperparameters, model_dict, model_file_name)


if __name__ == "__main__":
   main(sys.argv[:])