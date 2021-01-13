import os
import pathlib

def model_selection(selected_model):
    model_dict = initialize_model_selection_dictionary()

    if selected_model == 'resnet18':
        model_name = 'resnet18'
        model_dict[model_name] = True
        print("Resent18 -- Pre-trained")
        model_file_name = model_file_naming(model_name)
    elif selected_model == 'resnet34':
        model_name = 'resnet34'
        model_dict[model_name] = True
        print("Resent34 -- Pre-trained")
        model_file_name = model_file_naming(model_name)
    elif selected_model == 'resnet50':
        model_name = 'resnet50'
        model_dict[model_name] = True
        print("Resent50 -- Pre-trained")
        model_file_name = model_file_naming(model_name)
    elif selected_model == 'resnet50-modnet':
        model_name = 'resnet50-modnet'
        model_dict[model_name] = True
        print("Resent50-modnet -- Pre-trained")
        model_file_name = model_file_naming(model_name)
    else:
        print("Invalid Augmentation key: {}".format(selected_model))

    print("Model file name:", model_file_name)
    print("")
    return model_dict, model_file_name


def initialize_model_selection_dictionary():
    model_dict = dict([('resnet18', False),
                       ('resnet34', False),
                       ('resnet50', False),
                       ('resnet50-modnet', False)])

    return model_dict

def model_file_naming(model_name):
    cwd = pathlib.Path.cwd()
    model_save_path = cwd.joinpath("csci508_final", "saved_models_and_results", "saved_models")

    max_file_num = 0
    for file in os.listdir(model_save_path):
        if file.endswith(".pth"):
            file_strings = file.split("_")
            if model_name == file_strings[0]:
                file_num_string = file_strings[1]
                file_num_string = file_num_string.split(".")
                file_num_string = file_num_string[0]
                file_num = int(file_num_string[1:])
                if file_num > max_file_num:
                    max_file_num = int(file_num)
            
    model_file_name = model_name + "_v" + str(max_file_num+1)

    return model_file_name

