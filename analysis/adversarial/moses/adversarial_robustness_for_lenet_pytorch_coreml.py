import torch
import torchvision
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import coremltools
import numpy as np

from art.attacks.evasion import FastGradientMethod, BoundaryAttack
from art.estimators.classification import PyTorchClassifier

from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn as nn
import torch.optim as optim


def create_lenet():
    pytorch_model = nn.Sequential(
        nn.Conv2d(1, 6, 5, padding=2),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2),
        nn.Conv2d(6, 16, 5, padding=0),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2),
        nn.Flatten(),
        nn.Linear(400, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10)
    )
    return pytorch_model


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def predict_dl(pytorch_model, data):
    output = []
    y_pred = []
    for x in data:
        # x = x.T / 255
        x = x.reshape(1, 1, 28, 28)
        x = torch.tensor(x, dtype=torch.float)
        #x = x.cuda()
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
    batch_size = 128
    img_rows, img_cols = 28, 28

    T = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    train_data = torchvision.datasets.MNIST('mnist_data', train=True, download=True, transform=T)
    test_data = torchvision.datasets.MNIST('mnist_data', train=False, download=True, transform=T)

    x_test = to_numpy(test_data.data)
    y_test = to_numpy(test_data.targets)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    valloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return valloader, x_test, y_test


def generate_correctly_classified_data(valloader, y_test, pytorch_model, coreml_model, total_number=500):
    X_correctly_classifier = np.zeros(shape=(total_number, 1, 28, 28))
    y_correctly_classifier = np.zeros(shape=(total_number, ))
    counter = 0
    for i, (data, labels) in enumerate(valloader):
        labels = labels.numpy()
        for j in range(len(data)):
            sample = data[j].reshape(1, 1, 28, 28)
            x = torch.tensor(sample, dtype=torch.float)
            #x = x.cuda()
            data_ = np.zeros((500,1, 28,28))
            data_[0] = sample.numpy()
            torch_out = pytorch_model(x)
            split_ = str(coreml_model.get_spec().description.input[0]).split('\n')
            name_1 = split_[0].replace('name: "', '')
            name_1 = name_1.replace('"', '')

            coreml_output = coreml_model.predict({name_1: data_})
            ground_truth = labels[j]
            value, pred = torch.max(torch_out, 1)
            pred = pred.data.cpu()
            pytorch_prediction = pred.numpy()[0]
            #print('shape:    ',coreml_output['var_57'][0].shape)
            coreml_prediction = np.argmax(coreml_output['var_57'][0])
            #print(coreml_prediction)
            print(coreml_prediction, pytorch_prediction)
            if ground_truth == pytorch_prediction and ground_truth == coreml_prediction:
                X_correctly_classifier[counter] = sample[0]
                y_correctly_classifier[counter] = labels[j]
                counter += 1
                if counter == total_number:
                    return X_correctly_classifier, y_correctly_classifier


def generate_original_data(pytorch_model_path, coreml_model_path, original_file_path, model_names, valloader, Y_test):
    for model_name in model_names:
        pytorch_path = pytorch_model_path + model_name + ".pth"
        coreml_path = coreml_model_path + model_name + ".mlmodel"

        pytorch_model = torch.load(pytorch_path, map_location=device)
        coreml_model = coremltools.models.MLModel(coreml_path)

        X_correctly_classifier, y_correctly_classifier = generate_correctly_classified_data(valloader, Y_test,
                                                                                            pytorch_model, coreml_model)

        original_file_name = original_file_path + model_name + ".npz"
        np.savez(original_file_name, x=X_correctly_classifier, y=y_correctly_classifier)


def generate_adv_data(pytorch_model_path, adversarial_file_path, original_file_path, model_names, valloader, x_test,
                      type_attack="FGSM"):
    batch_size = 128
    img_rows, img_cols = 28, 28
    for model_name in model_names:
        pytorch_path = pytorch_model_path + model_name + ".pth"
        original_file_name = original_file_path + model_name + ".npz"

        pytorch_model = torch.load(pytorch_path, map_location=device)

        X_correctly_classifier, y_correctly_classifier = np.load(original_file_name)["x"], np.load(original_file_name)[
            "y"]

        # prepare attack
        cnn = create_lenet().to("cpu")
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(cnn.parameters(), lr=1e-3)
        optimizer = optim.SGD(cnn.parameters(), lr=1e-1, momentum=0.9)

        classifier = PyTorchClassifier(model=pytorch_model,
                                       clip_values=(np.min(x_test), np.max(x_test)),
                                       loss=criterion,
                                       optimizer=optimizer,
                                       input_shape=(1, img_rows, img_cols),
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

        split_ = str(coreml_model.get_spec().description.input[0]).split('\n')
        name_1 = split_[0].replace('name: "', '')
        name_1 = name_1.replace('"', '')

        coreml_output_fgsm = coreml_model.predict({name_1: X_adv_fgsm})
        coreml_predictions_fgsm = np.argmax(coreml_output_fgsm['var_57'], axis=1)
        coreml_output_boundary = coreml_model.predict({name_1: X_adv_boundary})
        coreml_predictions_boundary = np.argmax(coreml_output_boundary['var_57'], axis=1)

        ground_truth_fgsm = y_correctly_classified_fgsm
        ground_truth_boundary = y_correctly_classified_boundary

        misclassification_before_cov_fgsm = 100 * (np.sum(pytorch_predictions_fgsm != ground_truth_fgsm) / len(ground_truth_fgsm))
        misclassification_before_cov_boundary = 100 * (np.sum(pytorch_predictions_boundary != ground_truth_boundary) / len(ground_truth_boundary))

        misclassification_after_cov_fgsm = 100 * (np.sum(coreml_predictions_fgsm != ground_truth_fgsm) / len(ground_truth_fgsm))
        misclassification_after_cov_boundary = 100 * (np.sum(coreml_predictions_boundary != ground_truth_boundary) / len(ground_truth_boundary))

        conversion_divergence_fgsm = np.sum(coreml_predictions_fgsm != pytorch_predictions_fgsm)
        conversion_divergence_boundary = np.sum(coreml_predictions_boundary != pytorch_predictions_boundary)

        abs_err_fgsm = np.mean(np.absolute(pytorch_output_fgsm - coreml_output_fgsm['var_57']))
        abs_err_boundary = np.mean(np.absolute(pytorch_output_boundary - coreml_output_boundary['var_57']))

        results[i] = [misclassification_before_cov_fgsm, misclassification_after_cov_fgsm, conversion_divergence_fgsm,
                      abs_err_fgsm, misclassification_before_cov_boundary, misclassification_after_cov_boundary,
                      conversion_divergence_boundary, abs_err_boundary]

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    file_name = "LeNet5_pytorch_coreml.xlsx"
    results_df = pd.DataFrame(results)
    results_df.columns = ['misclassification_before_conv_fgsm', 'misclassification_after_conv_fgsm',
                          'conversion_divergence_fgsm', 'abs_err_fgsm', 'misclassification_before_cov_boundary',
                          'misclassification_after_cov_boundary', 'conversion_divergence_boundary', 'abs_err_boundary']
    writer = pd.ExcelWriter(file_name)

    # Write each dataframe to a different worksheet.
    results_df.to_excel(writer, sheet_name='LeNet5_coreml_coreml')

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


def main():
    np.random.seed(1001)
    torch.manual_seed(1001)

    valloader, x_test, y_test = prepare_data()

    #pytorch_model_path = "../../Training/pytorch/lenet5/"
    #coreml_model_path = "../../conversion/coremltools/pytorch/lenet5"
    #original_file_path = "../data/original/pytorch/lenet/"
    #adversarial_file_path = "../data/adversarial/pytorch/lenet/"
    pytorch_model_path = "/Volumes/Cisco/Fall2021/onnx-exchange/Training/pytorch/lenet5/"
    coreml_model_path = "/Volumes/Cisco/Fall2021/onnx-exchange/conversion/coremltools/pytorch/Lenet5/"
    original_file_path = "/Volumes/Cisco/Summer2022/onnx-exchange/adversarial/original/pytorch/lenet/"
    adversarial_file_path = "/Volumes/Cisco/Summer2022/onnx-exchange/adversarial/adversarial/pytorch/lenet/"

    model_names = ["torch_lenet5-mnist_2021-10-31_1", "torch_lenet5-mnist_2021-11-01_2", "torch_lenet5-mnist_2021-11-01_3",
                   "torch_lenet5-mnist_2021-11-01_4", "torch_lenet5-mnist_2021-11-01_5", "torch_lenet5-mnist_2021-11-01_6",
                   "torch_lenet5-mnist_2021-11-01_7", "torch_lenet5-mnist_2021-11-01_8", "torch_lenet5-mnist_2021-11-01_9",
                   "torch_lenet5-mnist_2021-11-01_10"]

    generate_original_data(pytorch_model_path, coreml_model_path, original_file_path, model_names, valloader, y_test)
    generate_adv_data(pytorch_model_path, adversarial_file_path, original_file_path, model_names, valloader, x_test,
                      type_attack="FGSM")
    generate_adv_data(pytorch_model_path, adversarial_file_path, original_file_path, model_names, valloader, x_test,
                      type_attack="Boundary")

    generate_results(pytorch_model_path, coreml_model_path, adversarial_file_path, original_file_path, model_names)


if __name__ == '__main__':
    main()
