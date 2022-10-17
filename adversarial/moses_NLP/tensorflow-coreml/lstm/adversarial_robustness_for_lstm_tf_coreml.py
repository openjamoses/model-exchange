import pandas as pd  # to load dataset
import numpy as np  # for mathematic equation
from tensorflow.keras.preprocessing.sequence import pad_sequences  # to do padding or truncating
import tensorflow as tf
import coremltools


def generate_tokens(tf_phrase, maximum_length, dataset):
    tf_phrase_list = tf_phrase.split()
    alzantot_tokens = []
    for x in tf_phrase_list:
        if x in dataset.dict.keys():
            alzantot_tokens.append(dataset.dict[x])
        else:
            alzantot_tokens.append(dataset.dict['UNK'])
    alzantot_tokens = pad_sequences([alzantot_tokens], maxlen=maximum_length, padding='post', truncating='post')
    return alzantot_tokens[0]


def transform_scores(scores):
    if scores.shape == (1,):
        return np.array([1 - scores[0], scores[0]])
    new_scores = np.zeros(shape=(len(scores), 2))

    for i in range(len(scores)):
        new_scores[i] = [1-scores[i], scores[i]]

    return new_scores


def get_max_length(x_train):
    review_length = []
    for review in x_train:
        review_length.append(len(review))

    return int(np.ceil(np.mean(review_length)))


def generate_results(tf_model_path, coreml_model_path, adversarial_file_path, model_names):

    results = np.zeros(shape=(10, 4))
    for i, model_name in enumerate(model_names):
        tf_path = tf_model_path + model_name + ".h5"
        coreml_path = coreml_model_path + model_name + ".mlmodel"
        adv_file_name = adversarial_file_path + model_name + "_alzantot.npz"

        tf_model = tf.keras.models.load_model(tf_path)
        coreml_model = coremltools.models.MLModel(coreml_path)

        X_adv, y_correctly_classified = np.load(adv_file_name)["x"], np.load(adv_file_name)["y"]

        tf_output = tf_model.predict(X_adv)
        tf_output = transform_scores(tf_output)
        tf_predictions = np.argmax(tf_output, axis=1)

        split_ = str(coreml_model.get_spec().description.input[0]).split('\n')
        name_1 = split_[0].replace('name: "', '')
        name_1 = name_1.replace('"', '')
        ort_outs = coreml_model.predict({name_1: X_adv})
        for key, val in ort_outs.items():
            print(key, val.shape)
        ort_outs = transform_scores(ort_outs['Identity'])
        coreml_predictions = np.argmax(ort_outs, axis=1)

        ground_truth = y_correctly_classified

        misclassification_before_cov = 100 * (np.sum(tf_predictions != ground_truth) / len(ground_truth))

        misclassification_after_cov = 100 * (np.sum(coreml_predictions != ground_truth) / len(ground_truth))

        conversion_divergence = np.sum(coreml_predictions != tf_predictions)

        abs_err = np.mean(np.absolute(tf_output - ort_outs))

        results[i] = [misclassification_before_cov, misclassification_after_cov, conversion_divergence, abs_err]

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    file_name = "lstm_tf_coreml.xlsx"
    results_df = pd.DataFrame(results)
    results_df.columns = ['misclassification_before_conv', 'misclassification_after_conv',
                          'conversion_divergence', 'abs_err']
    writer = pd.ExcelWriter(file_name)

    # Write each dataframe to a different worksheet.
    results_df.to_excel(writer, sheet_name='lstm_tf_coreml')

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


def main():

    tf_model_path = "/Volumes/Cisco/Fall2021/onnx-exchange/Training/Keras/lstm/"
    coreml_model_path = "/Volumes/Cisco/Fall2021/onnx-exchange/conversion/coremltools/keras/lstm/"
    adversarial_file_path = "/Volumes/Cisco/Summer2022/onnx-exchange/adversarial/moses_NLP/tensorflow-coreml/lstm/data/"

    model_names = ["tf_lstm-imdb_2021-10-29_1", "tf_lstm-imdb_2021-10-29_2", "tf_lstm-imdb_2021-10-30_3",
                   "tf_lstm-imdb_2021-10-30_4", "tf_lstm-imdb_2021-10-30_5", "tf_lstm-imdb_2021-10-30_6",
                   "tf_lstm-imdb_2021-10-30_7", "tf_lstm-imdb_2021-10-30_8", "tf_lstm-imdb_2021-10-31_9",
                   "tf_lstm-imdb_2021-10-31_10"]

    generate_results(tf_model_path, coreml_model_path, adversarial_file_path, model_names)


if __name__ == '__main__':
    main()
