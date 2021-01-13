import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import time
import pathlib
import csv
import os
import sys
import wandb

from .load_data import load_data
from .architectures.model_loader import model_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cwd = pathlib.Path.cwd()
model_save_path = cwd.joinpath("csci508_final", "saved_models_and_results", "saved_models")
model_results_save_path = cwd.joinpath("csci508_final", "saved_models_and_results", "saved_results")


def main(hyperparameters, model_dict, model_file_name):
    # Assuming that we are on a CUDA machine, this should print a CUDA device.
    print("Using device: ", device)
    print("")

    # Log for names of project, model, and W&B artifacts
    naming_log = dict(
        project_name = "Computer-Vision-Recyclable-Classifier",
        model_name = model_file_name)

    # Load Data
    print("Loading database of images ...")
    train_loader, val_loader, test_loader, class_names, naming_log, aug_dict = load_data(hyperparameters, 
                                                                                         naming_log)
    # visualize_training_data(train_loader, class_names)  # Optionally visualize some images
    num_classes = len(class_names)
    print("")

    # Create network.
    # model = Net(num_classes)
    model = model_loader(model_dict, num_classes)
    model.to(device)

    # Count the number of trainable parameters.
    print("Trainable model parameters:")
    for p in model.parameters():
        if p.requires_grad:
            print("Tensor ", p.shape, " number of params: ", p.numel())
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of trainable parameters: %d" % num_params)
    print("")

    # Run some random data through the network to check the output size.
    x = torch.zeros((2, 3, 256, 341)).to(device)  # minibatch size 2, image size [3, 256, 341]
    scores = model(x)
    print("Quick test to verify output size: should be [2, %d]:" % num_classes)
    print(scores.size())        # Should be size (2, num_classes)
    print("")

    # Collect Artifacts of Pre-Trained & Trained Model
    run_train = wandb.init(project=naming_log['project_name'], 
                                              name=naming_log['model_name']+"_train", 
                                              config=hyperparameters)
    run_train.use_artifact(naming_log['project_name'] + '/' + naming_log['artifact_Train_data'] + ":latest")
    run_train.use_artifact(naming_log['project_name'] + '/' + naming_log['artifact_Val_data'] + ":latest")
    naming_log['artifact_pretrained_model'] = naming_log['model_name']+"_untrained"
    model_artifact = wandb.Artifact(naming_log['artifact_pretrained_model'], 
                                    type="model", 
                                    description="Untrained: " + naming_log['model_name'],
                                    metadata=dict(wandb.config))
    run_train.log_artifact(model_artifact)

    # Train the network.
    print("Training the network ...")
    start_time = time.time()
    run_train, naming_log = train_model(model, 
                                        model_file_name, 
                                        train_loader, 
                                        val_loader, 
                                        hyperparameters['learn_rate'], 
                                        hyperparameters['epoch_qty'], 
                                        run_train,
                                        naming_log,
                                        class_names)
    print("Total training time: %f seconds" % (time.time()-start_time))
    print("")
    run_train.finish()

    # Log Model for Testing
    run_test = wandb.init(project=naming_log['project_name'], 
                                              name=naming_log['model_name']+"_test")
    run_test.use_artifact(naming_log['project_name'] + '/' + naming_log['artifact_trained_model'] + ":latest")
    run_test.use_artifact(naming_log['project_name'] + '/' + naming_log['artifact_Test_data'] + ":latest")
    naming_log['artifact_tested_model'] = naming_log['model_name']+"_tested"

    # Test the network.
    print('Evaluating accuracy on test set ...')
    confusion_matrix, run_test = eval_test_accuracy(test_loader, model, class_names, run_test)

    # Save all model performance metrix & hyperparameters to csv file on local computer
    write_to_file(confusion_matrix, 
                  model_file_name, 
                  class_names, 
                  aug_dict, 
                  hyperparameters)
    print("")

    # Show some example classifications.
    print("Results on example test images:")
    run_test = show_example_results(test_loader, model, class_names, run_test)
    print("")
    run_test.finish()

    print("All done!")


def visualize_training_data(loader, class_names):
    # get some random training images
    dataiter = iter(loader)
    images, labels = dataiter.next()

    # print class labels
    L = labels.numpy()
    out_string = ""
    for i in range(len(L)):
        out_string += "%s " % class_names[L[i]]

    print("")
    print("Classes of Images Shown:")
    out_string = out_string.split
    for image in range(len(out_string())):
        print("{}) {}".format(image+1, out_string()[image]))

    # show images
    imshow(torchvision.utils.make_grid(images))


# Show examples of images in loader
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()


def train_model(model, model_file_name, train_loader, val_loader, learning_rate, epochs, run_train, naming_log, class_names):
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    model = model.to(device=device)  # move the model parameters to CPU/GPU

    # Log gradients and model parameters for W&B
    run_train.watch(model)

    # Initialize lists for collecting model performance metrics during training
    train_loss = []
    epoch_val = []
    val_acc = []
    class_0_acc = []
    class_1_acc = []
    class_2_acc = []
    class_3_acc = []
    class_4_acc = []
    class_5_acc = []

    for e in range(epochs):
        print("Epoch: {0} / {1}".format(e+1, epochs))

        for t, (x, y) in enumerate(train_loader):
            model.train()  # put model to training mode
            x = x.to(device=device)  # move to device, e.g. GPU
            y = y.to(device=device)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % 100 == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                acc, class_acc = check_val_accuracy(val_loader, model, class_names)
        
        acc, class_acc = check_val_accuracy(val_loader, model, class_names)

        # Collect model performance metrics each epoch during training
        val_acc.append(acc)
        train_loss.append(loss)
        epoch_val.append(e+1)
        class_0_acc.append(class_acc[0])
        class_1_acc.append(class_acc[1])
        class_2_acc.append(class_acc[2])
        class_3_acc.append(class_acc[3])
        class_4_acc.append(class_acc[4])
        class_5_acc.append(class_acc[5])

    # Log Training Loss vs Epochs    
    data = [[x, y] for (x, y) in zip(epoch_val, train_loss)]
    table = wandb.Table(data=data, columns = ["Epoch", "Loss"])
    wandb.log({"Training Loss" : wandb.plot.line(table, "Epoch", "Loss", title="Training Loss")})

    # Log Validation Accuracy vs Epochs
    data = [[x, y] for (x, y) in zip(epoch_val, val_acc)]
    table = wandb.Table(data=data, columns = ["Epoch", "Accuracy"])
    wandb.log({"Validation Accuracy" : wandb.plot.line(table, 
                                                      "Epoch", 
                                                      "Accuracy", 
                                                      title="Total Validation Accuracy vs. Epoch")})

    # Log Class Validation Accuracy vs Epochs
    data0 = [[x, y] for (x, y) in zip(epoch_val, class_0_acc)]
    data1 = [[x, y] for (x, y) in zip(epoch_val, class_1_acc)]
    data2 = [[x, y] for (x, y) in zip(epoch_val, class_2_acc)]
    data3 = [[x, y] for (x, y) in zip(epoch_val, class_3_acc)]
    data4 = [[x, y] for (x, y) in zip(epoch_val, class_4_acc)]
    data5 = [[x, y] for (x, y) in zip(epoch_val, class_5_acc)]
    table0 = wandb.Table(data=data0, columns = ["Epoch", "Accuracy"])
    table1 = wandb.Table(data=data1, columns = ["Epoch", "Accuracy"])
    table2 = wandb.Table(data=data2, columns = ["Epoch", "Accuracy"])
    table3 = wandb.Table(data=data3, columns = ["Epoch", "Accuracy"])
    table4 = wandb.Table(data=data4, columns = ["Epoch", "Accuracy"])
    table5 = wandb.Table(data=data5, columns = ["Epoch", "Accuracy"])
    wandb.log({class_names[0] + " Validation Accuracy" : wandb.plot.line(table0, 
                                                       "Epoch", 
                                                       "Accuracy", 
                                                       title=class_names[0] + " Validation Accuracy vs. Epochs")})
    wandb.log({class_names[1] + " Validation Accuracy" : wandb.plot.line(table1, 
                                                       "Epoch", 
                                                       "Accuracy", 
                                                       title=class_names[1] + " Validation Accuracy vs. Epochs")})
    wandb.log({class_names[2] + " Validation Accuracy" : wandb.plot.line(table2, 
                                                       "Epoch", 
                                                       "Accuracy", 
                                                       title=class_names[2] + " Validation Accuracy vs. Epochs")})
    wandb.log({class_names[3] + " Validation Accuracy" : wandb.plot.line(table3, 
                                                       "Epoch", 
                                                       "Accuracy", 
                                                       title=class_names[3] + " Validation Accuracy vs. Epochs")})
    wandb.log({class_names[4] + " Validation Accuracy" : wandb.plot.line(table4, 
                                                       "Epoch", 
                                                       "Accuracy", 
                                                       title=class_names[4] + " Validation Accuracy vs. Epochs")})
    wandb.log({class_names[5] + " Validation Accuracy" : wandb.plot.line(table5, 
                                                       "Epoch", 
                                                       "Accuracy", 
                                                       title=class_names[5] + " Validation Accuracy vs. Epochs")})
    
    # Save model locally
    PATH = cwd.joinpath(model_save_path, model_file_name + ".pth")
    torch.save(model.state_dict(), PATH)
    print("Model saved locally! Model path is shown below:")
    print(PATH)
    print()

    # Save model to wandb
    PATH = os.path.join(wandb.run.dir, wandb.run.name + ".pth")
    torch.save(model.state_dict(), PATH)
    print("Model saved to wandb! Model path is shown below:")
    print(PATH)
    print()

    # save trained model as artifact
    naming_log['artifact_trained_model'] = naming_log['model_name']+"_trained"
    trained_model_artifact = wandb.Artifact(naming_log['artifact_trained_model'], 
                                            type="model",
                                            description="Trained: " + naming_log['model_name'],
                                            metadata=dict(wandb.config))
    trained_model_artifact.add_file(PATH)
    run_train.log_artifact(trained_model_artifact)

    return run_train, naming_log


def check_val_accuracy(val_loader, model, class_names):
    # Initialize lists for collecting model performance metrics during validation
    num_correct = 0
    num_samples = 0
    predictions = []
    ground_truth = []
    class_acc = []

    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        # Evaluate overall model accuracy during validation
        for x, y in val_loader:
            x = x.to(device=device)  # move to device, e.g. GPU
            y = y.to(device=device)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

            # Save predictions and truths to measure model accuracy for each class
            for truth, pred in zip(y.view(-1), preds.view(-1)):
                predictions.append(pred.item())
                ground_truth.append(truth.item())

        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%f) on validation' % (num_correct, num_samples, acc))
        print()

        # Evaluate model accuracy for each class during validation
        for i in range(0,len(class_names)):
            truth_count = 0
            pred_count = 0
            for j in range(0,len(ground_truth)):
                truth_val = ground_truth[j]
                pred_val = predictions[j]
                if truth_val == i:
                    truth_count = truth_count + 1
                    if pred_val == truth_val:
                        pred_count = pred_count + 1

            class_acc.append(pred_count/truth_count)
            
    return acc, class_acc


def eval_test_accuracy(test_loader, model, class_names, run_test):
    # Create the confusion matrix.
    num_classes = len(class_names)
    confusion_matrix = np.zeros((num_classes, num_classes))
    model.eval()  # set model to evaluation mode

    # Initialize lists for collecting model performance metrics during testing
    predictions = []
    ground_truth = []

    with torch.no_grad():
        # Test model
        for x, y in test_loader:
            x = x.to(device=device)  # move to device, e.g. GPU
            y = y.to(device=device)
            scores = model(x)
            
            # get the index of the max log-probability
            pred = scores.max(1, keepdim=True)[1]

            # Save predictions and truths to measure model accuracy for each class
            for t, p in zip(y.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
                predictions.append(p.item())
                ground_truth.append(t.item())

    # Print confusion matrix to terminal
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix.astype(int))

    # Log confution matrix to W&B project
    wandb.log({"conf_mat" : wandb.plot.confusion_matrix(predictions, ground_truth, class_names)})

    # Log model accuracy per class to W&B project
    print("Accuracy per class:")
    # This is the number of times a class was detected correctly, divided by
    # the total number of times that class was presented to the system.
    values = confusion_matrix.diagonal()/confusion_matrix.sum(1)
    labels = class_names
    data = [[label, val] for (label, val) in zip(labels, values)]
    table = wandb.Table(data=data, columns = ["Material Class", "Test Accuracy"])
    wandb.log({"Class Accuracies" : wandb.plot.bar(table, 
                                                   "Material Class", 
                                                   "Test Accuracy", 
                                                   title="Material Class Test Accuracy")})
    print(values)

    # Log overall model accuracy to W&B
    accuracy_score = np.sum(confusion_matrix.diagonal())/np.sum(confusion_matrix)
    print("Overall accuracy:")
    print(accuracy_score)
    wandb.log({"Test Accuracy": 100. * accuracy_score})

    return confusion_matrix, run_test


def write_to_file(confusion_matrix, model_file_name, class_names, transform_dict, hyperparameters):
    train_ratio = hyperparameters['train_ratio']
    val_ratio = hyperparameters['val_ratio']
    test_ratio = hyperparameters['test_ratio']
    epochs = hyperparameters['epoch_qty']

    # name of csv file
    file_name = cwd.joinpath(model_results_save_path, model_file_name + "_results.csv")

    class_accuracy = [confusion_matrix.diagonal() / confusion_matrix.sum(1)]
    overall_accuracy = np.sum(confusion_matrix.diagonal()) / np.sum(confusion_matrix)

    # writing to csv file
    with open(file_name, 'w', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # Write Model file name
        csvwriter.writerow(["Model file name:", model_file_name])
        csvwriter.writerows(' ')

        # Write Selected Tranforms
        csvwriter.writerow(["Selected Transforms:"])
        count = 1
        if transform_dict['horizontal']:
            csvwriter.writerow([str(count) +") Random Horizontal Flip"])
            count += 1
        if transform_dict['vertical']:
            csvwriter.writerow([str(count) +") Random Vertical Flip"])
            count += 1
        if transform_dict['rot30']:
            csvwriter.writerow([str(count) +") Random +/-30 Degree Rotation"])
            count += 1
        if transform_dict['noise']:
            csvwriter.writerow([str(count) +") Noise"])
            count += 1
        if transform_dict['blur']:
            csvwriter.writerow([str(count) +") Blur"])
            count += 1
        csvwriter.writerows(' ')

        # Write Data Split Info
        csvwriter.writerow(["Training Ratio:", round(train_ratio, 3)])
        csvwriter.writerow(["Validation Ratio:", round(val_ratio, 3)])
        csvwriter.writerow(["Testing Ratio:", round(test_ratio, 3)])
        csvwriter.writerows(' ')

        # Write Epoch Info
        csvwriter.writerow(["Epoch Quantity:", epochs])
        csvwriter.writerows(' ')
    
        # Write Confusion Matrix
        csvwriter.writerow(["Confusion Matrix"])
        csvwriter.writerow(class_names)
        csvwriter.writerows(confusion_matrix.astype(int))
        csvwriter.writerows(' ')

        # Write Class Accuracies
        csvwriter.writerow(["Class Accuracy"])
        csvwriter.writerow(class_names)
        csvwriter.writerows(class_accuracy)
        csvwriter.writerows(' ')

        # Write Overall Accuracy
        csvwriter.writerow(["Overall Accuracy:", overall_accuracy])
    
    print("Model results saved! Model results path is shown below:")
    print(file_name)


def show_example_results(loader, model, class_names, run_test):
    dataiter = iter(loader)
    images, labels = dataiter.next()

    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        x = images
        x = x.to(device=device)  # move to device, e.g. GPU
        scores = model(x)
        max_scores, preds = scores.max(1)

        # print true labels
        print("True labels:")
        L = labels.numpy()
        out_string = ""
        for i in range(len(L)):
            out_string += "%s " % class_names[L[i]]
        print(out_string)

        # print predicted labels
        print("Predicted labels:")
        out_string = ""
        for i in range(len(preds)):
            out_string += "%s " % class_names[preds[i].item()]
        print(out_string)

        # print scores
        print("Scores:")
        out_string = ""
        for i in range(len(max_scores)):
            out_string += "%.2f " % max_scores[i].item()
        print(out_string)

        # Log images in W&B account
        example_images = []
        for i in range(len(preds)):
            pred_class = class_names[preds[i].item()]
            true_class = class_names[L[i]]
            example_images.append(wandb.Image(images[i], caption="Pred: '{}' | Truth: '{}'".format(pred_class, true_class)))
    wandb.log({"Examples": example_images})
    # imshow(torchvision.utils.make_grid(images))
    return run_test


if __name__ == '__main__':
    sys.exit(main())
