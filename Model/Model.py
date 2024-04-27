import torch.nn as nn
import torch


class ADEP(nn.Module):
    def __init__(self, Input_size, label_size):
        super(ADEP, self).__init__()
        self.input_size = Input_size
        self.label_size = label_size
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, self.input_size),
            nn.Sigmoid()
        )

        self.discriminator = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048, 512 + 256),
            nn.BatchNorm1d(512 + 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512 + 256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.label_size),
            nn.LogSoftmax()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        classification = self.classifier(encoded)

        random_latent = torch.randn_like(encoded)

        real_output = self.discriminator(encoded)
        fake_output = self.discriminator(random_latent)

        return decoded, classification, real_output, fake_output


class AutoencoderClassifier(nn.Module):
    def __init__(self, Input_size, label_size):
        super(AutoencoderClassifier, self).__init__()
        self.input_size = Input_size
        self.label_size = label_size
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, self.input_size),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048, 512 + 256),
            nn.BatchNorm1d(512 + 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512 + 256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.label_size),
            nn.LogSoftmax()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        classification = self.classifier(encoded)

        return decoded, classification
