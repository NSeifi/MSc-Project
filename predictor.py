import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Predictor(nn.Module):
    def __init__(self, n_experiment_features = 7, n_hidden_layer = 32, n_out_labels = 1, encoder_layers = 2):
        super(Predictor, self).__init__()
        self.emb = nn.Linear(n_experiment_features, n_hidden_layer, bias=True)
        self.encoder = nn.LSTM(n_hidden_layer, n_hidden_layer, encoder_layers)
        self.generator = nn.Linear(n_hidden_layer, n_out_labels, bias=True)
        self.encoder_layers = encoder_layers
        self.encoder_bidirectional = False
        self.encoder_hidden = n_hidden_layer
        self.criterion = nn.MSELoss()

    def forward(self, input_tensor, target_tensor=None):
        """
        :param input_tensor: number of trade days * number of features
        :param target_tensor: number of target classes for that specific month
        :return: predicted classes for that month and the prediction loss value
        """
        e = self.emb(input_tensor).unsqueeze(1)
        init_context = self.encoder_init(1)
        encoded, output_context = self.encoder(e, init_context)
        pred = self.generator(output_context[0]).view(-1)
        loss = 0
        if target_tensor is not None:
            loss = self.criterion(pred, target_tensor)
        return pred, loss

    def encoder_init(self, batch_size):
        # num_layers * num_directions, batch, hidden_size
        return torch.zeros(self.encoder_layers * (2 if self.encoder_bidirectional else 1),
                           batch_size, self.encoder_hidden, device=device, dtype=torch.float32), \
               torch.zeros(self.encoder_layers * (2 if self.encoder_bidirectional else 1),
                           batch_size, self.encoder_hidden, device=device, dtype=torch.float32)