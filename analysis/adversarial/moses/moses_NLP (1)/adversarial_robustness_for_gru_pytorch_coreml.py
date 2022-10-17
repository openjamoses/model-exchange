import pandas as pd  # to load dataset
import numpy as np  # for mathematic equation
from nltk.corpus import stopwords  # to get collection of stopwords
import coremltools

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import string
import re
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import json

is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
# if is_cuda:
#     device = torch.device("cuda")
#     print("GPU is available")
# else:
#     device = torch.device("cpu")
#     print("GPU not available, CPU used")

device = torch.device("cpu")

def _flatten(object):
    flattened_list = []
    for item in object:
        if isinstance(item, (list, tuple)):
            flattened_list.extend(_flatten(item))
        else:
            flattened_list.append(item)
    return flattened_list


def convert_to_coreml_inputs(input_description, inputs):
    """Convenience function to combine a CoreML model's input description and
    set of raw inputs into the format expected by the model's predict function.
    """
    flattened_inputs = _flatten(inputs)
    coreml_inputs = {
        str(x): to_numpy(inp).astype(np.float64) for x, inp in zip(input_description, flattened_inputs)
    }
    #str(x): to_numpy(inp).numpy() for x, inp in zip(input_description, flattened_inputs)
    return coreml_inputs
from coremltools.converters.mil.mil import types
def _convert_to_inputtype(inputs):
    if isinstance(inputs, list):
        return [_convert_to_inputtype(x) for x in inputs]
    elif isinstance(inputs, tuple):
        return tuple([_convert_to_inputtype(x) for x in inputs])
        #return tuple([coremltools.TensorType(shape=x.shape) for x in inputs])
    elif isinstance(inputs, torch.Tensor):
        return coremltools.TensorType(shape=inputs.shape, dtype=types.int64)
    else:
        raise ValueError("Unable to parse type {} into InputType.".format(type(inputs)))


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def get_phrase_list(x_orig_pytorch, reverse_dict):
    phrase_list = []
    x_orig_pytorch = to_numpy(x_orig_pytorch)
    for idx in x_orig_pytorch:
        if idx != 0:
            phrase_list.append(reverse_dict[idx])
    return phrase_list


def generate_tokens(pytorch_phrase_list, dataset):
    alzantot_tokens = []
    for x in pytorch_phrase_list:
        if x in dataset.dict.keys():
            alzantot_tokens.append(dataset.dict[x])
        else:
            alzantot_tokens.append(dataset.dict['UNK'])
    alzantot_tokens = padding_([alzantot_tokens], 500)
    a = np.sum(np.sign(alzantot_tokens))
    return alzantot_tokens[0]


def transform_scores(scores):
    scores = to_numpy(scores)
    if scores.shape == (1,):
        return np.array([1 - scores[0], scores[0]])
    new_scores = np.zeros(shape=(len(scores), 2))

    for i in range(len(scores)):
        new_scores[i] = [1 - scores[i], scores[i]]

    return new_scores


def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)

    return s


def tockenize(x_train, y_train, x_val, y_val):
    word_list = []

    stop_words = set(stopwords.words('english'))
    for sent in x_train:
        for word in sent.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != '':
                word_list.append(word)

    corpus = Counter(word_list)
    # sorting on the basis of most common words
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[:1000]
    # creating a dict
    onehot_dict = {w: i + 1 for i, w in enumerate(corpus_)}
    reverse_dict = {i + 1: w for i, w in enumerate(corpus_)}

    # tockenize
    final_list_train, final_list_test = [], []
    for sent in x_train:
        final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                 if preprocess_string(word) in onehot_dict.keys()])
    for sent in x_val:
        final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                if preprocess_string(word) in onehot_dict.keys()])

    encoded_train = [1 if label == 'positive' else 0 for label in y_train]
    encoded_test = [1 if label == 'positive' else 0 for label in y_val]
    return np.array(final_list_train), np.array(encoded_train), np.array(final_list_test), np.array(
        encoded_test), onehot_dict, reverse_dict


def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features


class GRURNN(nn.Module):
    def __init__(self, no_layers, vocab_size, hidden_dim, embedding_dim, drop_prob=0.5, batch_size=100, output_dim=1):
        # super(SentimentRNN,self).__init__()
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.no_layers = no_layers
        self.vocab_size = vocab_size

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # lstm
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=self.hidden_dim,
                          num_layers=no_layers, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layer
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        # embeddings and lstm_out
        embeds = self.embedding(x)  # shape: B x S x Feature   since batch = True
        # print(embeds.shape)  #[50, 500, 1000]
        gru_out, hidden = self.gru(embeds, hidden)

        gru_out = gru_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully connected layer
        out = self.dropout(gru_out)
        out = self.fc(out)

        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)

        sig_out = sig_out[:, -1]  # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.no_layers,batch_size,self.hidden_dim).zero_().to(device)
        return hidden


def generate_results(pytorch_model_path, coreml_model_path, adversarial_file_path, model_names):
    results = np.zeros(shape=(10, 4))
    no_layers = 2
    vocab_size = 1000 + 1
    hidden_dim = 256
    embedding_dim = 64
    for i, model_name in enumerate(model_names):
        pytorch_path = pytorch_model_path + model_name + ".pb"
        coreml_path = coreml_model_path + model_name + ".mlmodel"
        adv_file_name = adversarial_file_path + model_name + "_alzantot.pt"

        pytorch_model = GRURNN(no_layers, vocab_size, hidden_dim, embedding_dim, drop_prob=0.5, batch_size=100)
        pytorch_model.load_state_dict(torch.load(pytorch_path, map_location=torch.device('cpu')))

        # moving to gpu
        pytorch_model.to(device)
        coreml_model = coremltools.models.MLModel(coreml_path)

        X_adv, y_correctly_classified = torch.load(adv_file_name)["x"], torch.load(adv_file_name)["y"]

        pytorch_output = np.zeros((1000, 2))
        ort_outs = np.zeros((1000, 2))
        ground_truth = np.array(y_correctly_classified)

        for idx in range(7):
            temp = torch.zeros((128, 500))
            # h0 = torch.zeros((no_layers, 300, hidden_dim)).to(device)
            # c0 = torch.zeros((no_layers, 300, hidden_dim)).to(device)
            weight = next(pytorch_model.parameters()).data
            hidden = weight.new(no_layers, 128, hidden_dim).zero_().to(device)
            if idx != 6:
                x = torch.reshape(X_adv[idx*128: (idx+1)*128], (128, 500))
                x = torch.tensor(x).to(device).long()
                # output, _ = pytorch_model(x, (h0, c0))
                output, _ = pytorch_model(x, hidden)
                output = transform_scores(output)
                pytorch_output[idx*128: (idx+1)*128] = output
            else:
                temp_torch = torch.zeros((128, 500))
                temp_torch[:104] = X_adv[896:]
                x = torch.tensor(temp_torch).to(device).long()
                # output, _ = pytorch_model(x, (h0, c0))
                output, _ = pytorch_model(x, hidden)
                output = transform_scores(output)
                pytorch_output[896:] = output[:104]

            # ort_inputs = {ort_session.get_inputs()[0].name: X_adv}
            # h0 = torch.zeros((no_layers, 300, hidden_dim)).to(device)
            # c0 = torch.zeros((no_layers, 300, hidden_dim)).to(device)
            weight = next(pytorch_model.parameters()).data
            hidden = weight.new(no_layers, 128, hidden_dim).zero_().to(device)
            if idx != 6:
                temp = X_adv[idx*128: (idx+1)*128]
                x = temp.to(device).long()
                # outs = ort_session.run(None, {ort_session.get_inputs()[0].name: to_numpy(x), 'h0': to_numpy(h0),
                #                                   'c0': to_numpy(c0)})
                # ort_outs = ort_session.run(None, ort_inputs)

                input_x = (x, hidden)
                coreml_inputs = convert_to_coreml_inputs(coreml_model.input_description, input_x)
                outs = coreml_model.predict(coreml_inputs)

                outs = outs['var_51']
                temp_outs = np.zeros((128, 2))
                temp_outs[:, 0] = 1 - outs
                temp_outs[:, 1] = outs
                # outs = np.array([1 - outs[0], outs[0]])
                # outs = transform_scores(outs[0][0])
                ort_outs[idx*128: (idx+1)*128] = temp_outs
            else:
                temp[:104] = X_adv[896:]
                x = torch.tensor(temp).to(device).long()
                # outs = ort_session.run(None, {ort_session.get_inputs()[0].name: to_numpy(x), 'h0': to_numpy(h0),
                #                               'c0': to_numpy(c0)})
                input_x = (x, hidden)
                coreml_inputs = convert_to_coreml_inputs(coreml_model.input_description, input_x)
                outs = coreml_model.predict(coreml_inputs)
                outs = outs['var_51']
                temp_outs = np.zeros((104, 2))
                temp_outs[:, 0] = 1 - outs[:104]
                temp_outs[:, 1] = outs[:104]
                # outs = np.array([1 - outs[0][:100], outs[0][:100]])
                ort_outs[896:] = temp_outs

        coreml_predictions = np.argmax(ort_outs, axis=1)
        pytorch_predictions = np.argmax(pytorch_output, axis=1)


        misclassification_before_cov = 100 * (np.sum(pytorch_predictions != ground_truth) / len(ground_truth))

        misclassification_after_cov = 100 * (np.sum(coreml_predictions != ground_truth) / len(ground_truth))

        conversion_divergence = np.sum(coreml_predictions != pytorch_predictions)

        abs_err = np.mean(np.absolute(pytorch_output - ort_outs))

        results[i] = [misclassification_before_cov, misclassification_after_cov, conversion_divergence, abs_err]

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    file_name = "gru_pytorch_coreml.xlsx"
    results_df = pd.DataFrame(results)
    results_df.columns = ['misclassification_before_conv', 'misclassification_after_conv',
                          'conversion_divergence', 'abs_err']
    writer = pd.ExcelWriter(file_name)

    # Write each dataframe to a different worksheet.
    results_df.to_excel(writer, sheet_name='gru_pytorch_coreml')

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


def main():
    np.random.seed(1001)

    pytorch_model_path = "/Volumes/Cisco/Summer2022/onnx-exchange/Train2/pytorch/gru/"
    coreml_model_path = "/Volumes/Cisco/Summer2022/onnx-exchange/Train2/conversion/coremltools/pytorch/gru/"
    adversarial_file_path = "/Volumes/Cisco/Summer2022/onnx-exchange/adversarial/moses/data/"

    model_names = ["torch_state_GRU-IMDb_2022-03-11_1", "torch_state_GRU-IMDb_2022-03-11_2",
                   "torch_state_GRU-IMDb_2022-03-11_3", "torch_state_GRU-IMDb_2022-03-11_4",
                   "torch_state_GRU-IMDb_2022-03-11_5", "torch_state_GRU-IMDb_2022-03-11_6",
                   "torch_state_GRU-IMDb_2022-03-11_7", "torch_state_GRU-IMDb_2022-03-11_8",
                   "torch_state_gru-imdb_2022-03-11_9", "torch_state_GRU-IMDb_2022-03-11_10"]

    generate_results(pytorch_model_path, coreml_model_path, adversarial_file_path, model_names)


if __name__ == '__main__':
    main()
