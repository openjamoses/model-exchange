import tensorflow as tf
import numpy as np
import pandas as pd
import coremltools

from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod, BoundaryAttack

tf.compat.v1.disable_eager_execution()


def prepare_data(nb_classes=10):
    img_rows, img_cols = 28, 28
    # Load image data with labels, split into test and training set
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # reshape images in 4D tensor (N images, 28 rows, 28 columns, 1 channel)
    # rescale pixels range from [0, 255] to [0, 1]
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255
    X_test /= 255
    print('X_train shape: ', X_train.shape)
    print(X_train.shape[0], "training samples")
    print(X_test.shape[0], "test samples")

    # convert digit labels (0-9) in one-hot encoded binary vectors.
    # These correspond to the training/test labels at the output of the net.
    Y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
    Y_test = tf.keras.utils.to_categorical(y_test, nb_classes)

    return X_train, Y_train, X_test, Y_test


def generate_correctly_classified_data(X_test, Y_test, tf_model, coreml_model, total_number=1000):
    random_ids = np.random.choice(len(Y_test), total_number * 5, replace=False)
    X_correctly_classifier = np.zeros(shape=(total_number, 28, 28, 1))
    y_correctly_classifier = np.zeros(shape=(total_number, 10))
    counter = 0
    for i in random_ids:
        sample = X_test[i].reshape(1, 28, 28, 1)
        tf_output = tf_model.predict(sample)
        coreml_output = coreml_model.predict({'conv2d_input': sample})
        ground_truth = np.argmax(Y_test[i])
        tf_prediction = np.argmax(tf_output, axis=1)[0]
        coreml_prediction = np.argmax(coreml_output['Identity'], axis=1)[0]

        if ground_truth == tf_prediction and ground_truth == coreml_prediction:
            X_correctly_classifier[counter] = X_test[i]
            y_correctly_classifier[counter] = Y_test[i]
            counter += 1
            if counter == total_number:
                return X_correctly_classifier, y_correctly_classifier


def generate_original_data(tf_model_path, coreml_model_path, original_file_path, model_names, X_test, Y_test):
    for model_name in model_names:
        tf_path = tf_model_path + model_name + ".h5"
        coreml_path = coreml_model_path + model_name + ".mlmodel"

        tf_model = tf.keras.models.load_model(tf_path)
        coreml_model = coremltools.models.MLModel(coreml_path)

        X_correctly_classifier, y_correctly_classifier = generate_correctly_classified_data(X_test, Y_test, tf_model,
                                                                                            coreml_model)

        original_file_name = original_file_path + model_name + ".npz"
        np.savez(original_file_name, x=X_correctly_classifier, y=y_correctly_classifier)


def generate_adv_data(tf_model_path, adversarial_file_path, original_file_path, model_names, X_test,
                      type_attack="FGSM"):
    for model_name in model_names:
        tf_path = tf_model_path + model_name + ".h5"
        original_file_name = original_file_path + model_name + ".npz"

        tf_model = tf.keras.models.load_model(tf_path)

        X_correctly_classifier, y_correctly_classifier = np.load(original_file_name)["x"], np.load(original_file_name)[
            "y"]

        # prepare attack
        classifier = KerasClassifier(model=tf_model, clip_values=(np.min(X_test), np.max(X_test)))
        attack = None
        if type_attack == "FGSM":
            attack = FastGradientMethod(estimator=classifier, eps=0.3)
        elif type_attack == "Boundary":
            attack = BoundaryAttack(estimator=classifier, targeted=False, max_iter=50)

        X_test_adv = attack.generate(X_correctly_classifier)

        outputs = tf_model.predict(X_test_adv)
        predictions = np.argmax(outputs, axis=1)
        ground_truth = np.argmax(y_correctly_classifier, axis=1)
        mis_rate = np.sum(predictions != ground_truth) / 10
        accuracy = np.sum(predictions == ground_truth) / 10
        print('mis_rate', mis_rate)
        print('accuracy', accuracy)

        adv_file_name = adversarial_file_path + model_name + "_" + type_attack + ".npz"
        np.savez(adv_file_name, x=X_test_adv, y=y_correctly_classifier)


def generate_results(tf_model_path, coreml_model_path, adversarial_file_path, original_file_path, model_names):

    results = np.zeros(shape=(10, 8))
    for i, model_name in enumerate(model_names):
        tf_path = tf_model_path + model_name + ".h5"
        coreml_path = coreml_model_path + model_name + ".mlmodel"
        adv_file_name_fgsm = adversarial_file_path + model_name + "_FGSM.npz"
        adv_file_name_boundary = adversarial_file_path + model_name + "_Boundary.npz"

        tf_model = tf.keras.models.load_model(tf_path)
        coreml_model = coremltools.models.MLModel(coreml_path)

        X_adv_fgsm, y_correctly_classified_fgsm = np.load(adv_file_name_fgsm)["x"], np.load(adv_file_name_fgsm)["y"]
        X_adv_boundary, y_correctly_classified_boundary = np.load(adv_file_name_boundary)["x"], \
                                                          np.load(adv_file_name_boundary)["y"]

        tf_output_fgsm = tf_model.predict(X_adv_fgsm)
        tf_predictions_fgsm = np.argmax(tf_output_fgsm, axis=1)
        tf_output_boundary = tf_model.predict(X_adv_boundary)
        tf_predictions_boundary = np.argmax(tf_output_boundary, axis=1)

        coreml_output_fgsm = coreml_model.predict({'conv2d_input': X_adv_fgsm})
        coreml_predictions_fgsm = np.argmax(coreml_output_fgsm['Identity'], axis=1)
        coreml_output_boundary = coreml_model.predict({'conv2d_input': X_adv_boundary})
        coreml_predictions_boundary = np.argmax(coreml_output_boundary['Identity'], axis=1)

        ground_truth_fgsm = np.argmax(y_correctly_classified_fgsm, axis=1)
        ground_truth_boundary = np.argmax(y_correctly_classified_boundary, axis=1)

        misclassification_before_cov_fgsm = 100 * (np.sum(tf_predictions_fgsm != ground_truth_fgsm) / len(ground_truth_fgsm))
        misclassification_before_cov_boundary = 100 * (np.sum(tf_predictions_boundary != ground_truth_boundary) / len(ground_truth_boundary))

        misclassification_after_cov_fgsm = 100 * (np.sum(coreml_predictions_fgsm != ground_truth_fgsm) / len(ground_truth_fgsm))
        misclassification_after_cov_boundary = 100 * (np.sum(coreml_predictions_boundary != ground_truth_boundary) / len(ground_truth_boundary))

        conversion_divergence_fgsm = np.sum(coreml_predictions_fgsm != tf_predictions_fgsm)
        conversion_divergence_boundary = np.sum(coreml_predictions_boundary != tf_predictions_boundary)

        abs_err_fgsm = np.mean(np.absolute(tf_output_fgsm - coreml_output_fgsm['Identity']))
        abs_err_boundary = np.mean(np.absolute(tf_output_boundary - coreml_output_boundary['Identity']))

        results[i] = [misclassification_before_cov_fgsm, misclassification_after_cov_fgsm, conversion_divergence_fgsm,
                      abs_err_fgsm, misclassification_before_cov_boundary, misclassification_after_cov_boundary,
                      conversion_divergence_boundary, abs_err_boundary]

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    file_name = "LeNet5_tf_coreml.xlsx"
    results_df = pd.DataFrame(results)
    results_df.columns = ['misclassification_before_conv_fgsm', 'misclassification_after_conv_fgsm',
                          'conversion_divergence_fgsm', 'abs_err_fgsm', 'misclassification_before_cov_boundary',
                          'misclassification_after_cov_boundary', 'conversion_divergence_boundary', 'abs_err_boundary']
    writer = pd.ExcelWriter(file_name)

    # Write each dataframe to a different worksheet.
    results_df.to_excel(writer, sheet_name='LeNet5_tf_coreml')

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


def main():
    np.random.seed(1001)
    tf.random.set_seed(1001)
    batch_size = 500  # Number of images processed at once
    nb_classes = 10  # 10 Digits from 0 to 9

    _, _, X_test, Y_test = prepare_data()


    tf_model_path = "/Volumes/Cisco/Fall2021/onnx-exchange/Training/Keras/lenet5/"
    coreml_model_path = "/Volumes/Cisco/Fall2021/onnx-exchange/conversion/coremltools/keras/Lenet5/"
    original_file_path = "/Volumes/Cisco/Summer2022/onnx-exchange/adversarial/original/Tensorflow/lenet/"
    adversarial_file_path = "/Volumes/Cisco/Summer2022/onnx-exchange/adversarial/adversarial/tensorflow/lenet/"

    model_names = ["tf_Lenet5_mnist_2021-10-27_1", "tf_Lenet5_mnist_2021-10-27_2", "tf_Lenet5_mnist_2021-10-27_3",
                   "tf_Lenet5_mnist_2021-10-27_4", "tf_Lenet5_mnist_2021-10-28_5", "tf_Lenet5_mnist_2021-10-28_6",
                   "tf_Lenet5_mnist_2021-10-28_7", "tf_Lenet5_mnist_2021-10-28_8", "tf_Lenet5_mnist_2021-10-28_9",
                   "tf_Lenet5_mnist_2021-10-28_10"]

    # model_names = ["tf_Lenet5_mnist_2021-10-28_10"]

    generate_original_data(tf_model_path, coreml_model_path, original_file_path, model_names, X_test, Y_test)
    generate_adv_data(tf_model_path, adversarial_file_path, original_file_path, model_names, X_test,
                      type_attack="FGSM")
    generate_adv_data(tf_model_path, adversarial_file_path, original_file_path, model_names, X_test,
                      type_attack="Boundary")
    generate_results(tf_model_path, coreml_model_path, adversarial_file_path, original_file_path, model_names)


if __name__ == '__main__':
    main()
