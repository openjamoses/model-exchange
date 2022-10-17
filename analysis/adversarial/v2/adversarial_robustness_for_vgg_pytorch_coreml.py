import torch
import torchvision
import matplotlib.pyplot as plt
import time
import os
import coremltools
import numpy as np
import pandas as pd

from art.attacks.evasion import FastGradientMethod, BoundaryAttack
from art.estimators.classification import PyTorchClassifier
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn as nn
import torch.optim as optim
from torchvision import models


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def predict_dl(pytorch_model, data):
    y_pred = []
    output = []
    for x in data:
        # x = x.T / 255
        x = x.reshape(1, 3, 32, 32)
        x = torch.tensor(x, dtype=torch.float)
        x = x.cuda()
        torch_out = pytorch_model(x)
        value, pred = torch.max(torch_out, 1)
        pred = pred.data.cpu()
        output.extend(list(to_numpy(torch_out)))
        y_pred.extend(list(pred.numpy()))
    return np.array(output), np.array(y_pred)


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("No Cuda Available")
print(device)


def prepare_data():
    num_classes = 10
    img_rows, img_cols = 32, 32
    channels = 3
    batch_size = 128
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_data = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]), download=True)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))

    valloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    x_test = np.array(test_data.data)
    y_test = np.array(test_data.targets)
    return valloader, x_test, y_test


def generate_correctly_classified_data(valloader, y_test, pytorch_model, coreml_model, total_number=1000):
    X_correctly_classifier = np.zeros(shape=(total_number, 3, 32, 32))
    y_correctly_classifier = np.zeros(shape=(total_number, ))
    counter = 0
    for i, (data, labels) in enumerate(valloader):
        labels = labels.numpy()
        for j in range(len(data)):
            sample = data[j].reshape(1, 3, 32, 32)
            x = torch.tensor(sample, dtype=torch.float)
            x = x.cuda()
            torch_out = pytorch_model(x)
            coreml_output = coreml_model.predict({'conv2d_input': sample.numpy()})
            ground_truth = labels[j]
            value, pred = torch.max(torch_out, 1)
            pred = pred.data.cpu()
            pytorch_prediction = pred.numpy()[0]
            coreml_prediction = np.argmax(coreml_output['Identity'], axis=1)[0]

            if ground_truth == pytorch_prediction and ground_truth == coreml_prediction:
                X_correctly_classifier[counter] = sample[0]
                y_correctly_classifier[counter] = labels[j]
                counter += 1
                if counter == total_number:
                    return X_correctly_classifier, y_correctly_classifier


def generate_original_data(pytorch_model_path, coreml_model_path, original_file_path, model_names, valloader, y_test):
    for model_name in model_names:
        pytorch_path = pytorch_model_path + model_name + ".pth"
        coreml_path = coreml_model_path + model_name + ".mlmodel"

        pytorch_model = torch.load(pytorch_path, map_location=device)
        coreml_model = coremltools.models.MLModel(coreml_path)

        X_correctly_classifier, y_correctly_classifier = generate_correctly_classified_data(valloader, y_test,
                                                                                            pytorch_model, coreml_model)

        original_file_name = original_file_path + model_name + ".npz"
        np.savez(original_file_name, x=X_correctly_classifier, y=y_correctly_classifier)


def generate_adv_data(pytorch_model_path, adversarial_file_path, original_file_path, model_names, valloader, x_test,
                      type_attack="FGSM"):
    batch_size = 128
    img_rows, img_cols, channels = 32, 32, 3
    for model_name in model_names:
        pytorch_path = pytorch_model_path + model_name + ".pth"
        original_file_name = original_file_path + model_name + ".npz"

        pytorch_model = torch.load(pytorch_path, map_location=device)

        X_correctly_classifier, y_correctly_classifier = np.load(original_file_name)["x"], np.load(original_file_name)[
            "y"]

        # prepare attack
        model_ft, input_size = initialize_model("vgg", num_classes=10, feature_extract=False, use_pretrained=True)

        cnn = model_ft.to("cpu")
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

        classifier = PyTorchClassifier(model=pytorch_model,
                                       clip_values=(np.min(x_test), np.max(x_test)),
                                       loss=criterion,
                                       optimizer=optimizer,
                                       input_shape=(channels, img_rows, img_cols),
                                       nb_classes=10,
                                       )

        attack = None
        if type_attack == "FGSM":
            attack = FastGradientMethod(estimator=classifier, eps=0.3)
        elif type_attack == "Boundary":
            attack = BoundaryAttack(estimator=classifier, targeted=False, max_iter=50)

        data = torch.tensor(X_correctly_classifier, dtype=torch.float)
        data = data.numpy()
        X_test_adv = attack.generate(data)

        adv_file_name = adversarial_file_path + model_name + "_" + type_attack + ".npz"
        np.savez(adv_file_name, x=X_test_adv, y=y_correctly_classifier)


def generate_results(pytorch_model_path, coreml_model_path, adversarial_file_path, original_file_path, model_names):

    results = np.zeros(shape=(10, 8))
    for i, model_name in enumerate(model_names):
        pytorch_path = pytorch_model_path + model_name + ".pth"
        coreml_path = coreml_model_path + model_name + ".mlmodel"
        original_file_name = original_file_path + model_name + ".npz"
        adv_file_name_fgsm = adversarial_file_path + model_name + "_FGSM.npz"
        adv_file_name_boundary = adversarial_file_path + model_name + "_Boundary.npz"

        pytorch_model = torch.load(pytorch_path, map_location=device)
        coreml_model = coremltools.models.MLModel(coreml_path)

        X_adv_fgsm, y_correctly_classified_fgsm = np.load(adv_file_name_fgsm)["x"], np.load(adv_file_name_fgsm)["y"]
        X_adv_boundary, y_correctly_classified_boundary = np.load(adv_file_name_boundary)["x"], \
                                                          np.load(adv_file_name_boundary)["y"]

        pytorch_output_fgsm, pytorch_predictions_fgsm = predict_dl(pytorch_model, X_adv_fgsm)
        pytorch_output_boundary, pytorch_predictions_boundary = predict_dl(pytorch_model, X_adv_boundary)

        coreml_output_fgsm = coreml_model.predict({'conv2d_input': X_adv_fgsm})
        coreml_predictions_fgsm = np.argmax(coreml_output_fgsm['Identity'], axis=1)
        coreml_output_boundary = coreml_model.predict({'conv2d_input': X_adv_boundary})
        coreml_predictions_boundary = np.argmax(coreml_output_boundary['Identity'], axis=1)

        ground_truth_fgsm = y_correctly_classified_fgsm
        ground_truth_boundary = y_correctly_classified_boundary

        misclassification_before_cov_fgsm = 100 * (np.sum(pytorch_predictions_fgsm != ground_truth_fgsm) / len(ground_truth_fgsm))
        misclassification_before_cov_boundary = 100 * (np.sum(pytorch_predictions_boundary != ground_truth_boundary) / len(ground_truth_boundary))

        misclassification_after_cov_fgsm = 100 * (np.sum(coreml_predictions_fgsm != ground_truth_fgsm) / len(ground_truth_fgsm))
        misclassification_after_cov_boundary = 100 * (np.sum(coreml_predictions_boundary != ground_truth_boundary) / len(ground_truth_boundary))

        conversion_divergence_fgsm = np.sum(coreml_predictions_fgsm != pytorch_predictions_fgsm)
        conversion_divergence_boundary = np.sum(coreml_predictions_boundary != pytorch_predictions_boundary)

        abs_err_fgsm = np.mean(np.absolute(pytorch_output_fgsm - coreml_output_fgsm['Identity']))
        abs_err_boundary = np.mean(np.absolute(pytorch_output_boundary - coreml_output_boundary['Identity']))

        results[i] = [misclassification_before_cov_fgsm, misclassification_after_cov_fgsm, conversion_divergence_fgsm,
                      abs_err_fgsm, misclassification_before_cov_boundary, misclassification_after_cov_boundary,
                      conversion_divergence_boundary, abs_err_boundary]

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    file_name = "vgg_pytorch_onnx.xlsx"
    results_df = pd.DataFrame(results)
    results_df.columns = ['misclassification_before_conv_fgsm', 'misclassification_after_conv_fgsm',
                          'conversion_divergence_fgsm', 'abs_err_fgsm', 'misclassification_before_cov_boundary',
                          'misclassification_after_cov_boundary', 'conversion_divergence_boundary', 'abs_err_boundary']
    writer = pd.ExcelWriter(file_name)

    # Write each dataframe to a different worksheet.
    results_df.to_excel(writer, sheet_name='vgg_pytorch_onnx')

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


def main():
    np.random.seed(1001)
    torch.manual_seed(1001)

    valloader, x_test, y_test = prepare_data()

    pytorch_model_path = "../../Training/pytorch/vgg/"
    onnx_model_path = "../../conversion/onnx/pytorch/vgg/"
    original_file_path = "../data/original/pytorch/vgg16/"
    adversarial_file_path = "../data/adversarial/pytorch/vgg16/"

    model_names = ["torch_exp_vgg_2021-11-03_1", "torch_exp_vgg_2021-11-03_2", "torch_exp_vgg_2021-11-03_3",
                   "torch_exp_vgg_2021-11-03_4", "torch_exp_vgg_2021-11-03_5", "torch_exp_vgg_2021-11-04_6",
                   "torch_exp_vgg_2021-11-04_7", "torch_exp_vgg_2021-11-04_8", "torch_exp_vgg_2021-11-04_9",
                   "torch_exp_vgg_2021-11-04_10"]

    generate_original_data(pytorch_model_path, onnx_model_path, original_file_path, model_names, valloader, y_test)
    generate_adv_data(pytorch_model_path, adversarial_file_path, original_file_path, model_names, valloader, x_test,
                      type_attack="FGSM")
    generate_adv_data(pytorch_model_path, adversarial_file_path, original_file_path, model_names, valloader, x_test,
                      type_attack="Boundary")
    generate_results(pytorch_model_path, onnx_model_path, adversarial_file_path, original_file_path, model_names)


if __name__ == '__main__':
    main()