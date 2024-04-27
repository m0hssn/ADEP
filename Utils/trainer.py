import torch
from metrica import Metrica
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm
import torch.nn as nn
from Model.Model import ADEP
from Model.Model import AutoencoderClassifier
from Model.Model import Autoencoder
from DataLoading import DDIDataset
import numpy as np

def train_ADEP(model, train_loader, test_loader, label_size, decoder_criterion, classifier_criterion,
               adversarial_criterion,
               optimizer, scheduler, num_epochs, best_acc, alpha, beta, gamma):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_accuracy = best_acc
    model = model.to(device)
    test_metrica = None

    for epoch in range(num_epochs):
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (training)", leave=False)
        running_loss = 0.0
        running_decoder_loss = 0.0
        running_classifier_loss = 0.0
        running_correct = 0
        running_total = 0

        test_metrica = Metrica(num_classes=label_size)

        for i, (inputs, labels) in enumerate(train_pbar):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            reconstructed_inputs, predicted_labels, real_output, fake_output = model(inputs)
            decoder_loss = decoder_criterion(reconstructed_inputs, inputs)
            classifier_loss = classifier_criterion(predicted_labels, labels)
            real_labels = torch.ones_like(real_output)
            fake_labels = torch.zeros_like(fake_output)
            adversarial_loss = adversarial_criterion(real_output, real_labels) + adversarial_criterion(fake_output,
                                                                                                       fake_labels)

            loss = alpha * decoder_loss + gamma * classifier_loss + beta * adversarial_loss

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_decoder_loss += decoder_loss.item()
            running_classifier_loss += classifier_loss.item()
            _, predicted = torch.max(predicted_labels.data, 1)
            running_total += labels.size(0)
            running_correct += (predicted == labels).sum().item()
            train_pbar.set_postfix({"loss": running_loss / (i + 1),
                                    "decoder_loss": running_decoder_loss / (i + 1),
                                    "classifier_loss": running_classifier_loss / (i + 1),
                                    "accuracy": 100 * running_correct / running_total})

            optimizer.zero_grad()
            real_loss = adversarial_criterion(real_output, real_labels)
            fake_loss = adversarial_criterion(fake_output, fake_labels)
            discriminator_loss = real_loss + fake_loss

            discriminator_loss.backward()
            optimizer.step()

        epoch_loss = running_loss / len(train_loader)
        epoch_decoder_loss = running_decoder_loss / len(train_loader)
        epoch_classifier_loss = running_classifier_loss / len(train_loader)
        epoch_train_accuracy = 100 * running_correct / running_total
        train_pbar.set_postfix({"loss": epoch_loss,
                                "decoder_loss": epoch_decoder_loss,
                                "classifier_loss": epoch_classifier_loss,
                                "accuracy": epoch_train_accuracy})

        model.eval()
        test_running_loss = 0.0
        test_running_correct = 0
        test_running_total = 0
        test_pbar = tqdm(test_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (testing)", leave=False)
        with torch.no_grad():
            for inputs, labels in test_pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                reconstructed_inputs, predicted_labels, _, _ = model(inputs)

                test_metrica.update(predicted_labels, labels)

                decoder_loss = decoder_criterion(reconstructed_inputs, inputs)
                classifier_loss = classifier_criterion(predicted_labels, labels)
                loss = decoder_loss + classifier_loss
                test_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(predicted_labels.data, 1)
                test_running_total += labels.size(0)
                test_running_correct += (predicted == labels).sum().item()
            test_accuracy = 100 * test_running_correct / test_running_total
            test_loss = test_running_loss / len(test_loader.dataset)
        test_pbar.set_postfix({"loss": test_loss,
                               "accuracy": test_accuracy})
        scheduler.step(test_accuracy)
        print(
            'Epoch [{}/{}], Train Loss: {:.4f}, Train Autoencoder Loss: {:.4f}, Train Classification Loss: {:.4f}, Train Accuracy: {:.2f}%, Test Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(
                epoch + 1, num_epochs, epoch_loss, epoch_decoder_loss, epoch_classifier_loss, epoch_train_accuracy,
                test_loss, test_accuracy))

        test_metrica.print_metrics()
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            model_scripted = torch.jit.script(model)
            model_scripted.save("best_model_ADEP.pt")

    return best_accuracy, test_metrica


class ADEPTrainer:
    def __init__(self, df_drug, extraction, Input_size, label_size, num_folds=5, batch_size=64, lr=3e-4, num_epochs=30,
                 index=1):
        self.test_loader = None
        self.df_drug = df_drug
        self.extraction = extraction
        self.Input_size = Input_size
        self.label_size = label_size
        self.num_folds = num_folds
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.best_acc = 0
        self.index = index
        self.metrica = 0

    def train(self):
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        fold = 1
        for train_index, test_index in kf.split(self.extraction):
            if fold == self.index:
                train_extraction = self.extraction.iloc[train_index]
                test_extraction = self.extraction.iloc[test_index]

                train_dataset = DDIDataset(self.df_drug, train_extraction)
                test_dataset = DDIDataset(self.df_drug, test_extraction)
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
                model = ADEP(self.Input_size, self.label_size)
                decoder_criterion = nn.L1Loss()
                adversarial_criterion = nn.BCELoss()
                classifier_criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,
                                                                       threshold=0.01, cooldown=2)
                self.test_loader = test_loader
                print("Training fold {}...".format(fold))
                self.best_acc, self.metrica = train_ADEP(model, train_loader, test_loader, self.label_size,
                                                         decoder_criterion, classifier_criterion,
                                                         adversarial_criterion, optimizer, scheduler,
                                                         self.num_epochs, self.best_acc, 0.5, 1, 1)

            fold += 1


def train_autoencoder_classifier(model, train_loader, test_loader, label_size, decoder_criterion, classifier_criterion,
                                 optimizer,
                                 scheduler, num_epochs, best_acc, alpha, gamma):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    best_accuracy = best_acc
    model = model.to(device)

    test_metrica = None

    for epoch in range(num_epochs):
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (training)", leave=False)
        running_loss = 0.0
        running_decoder_loss = 0.0
        running_classifier_loss = 0.0
        running_correct = 0
        running_total = 0

        test_metrica = Metrica(num_classes=label_size)

        for i, (inputs, labels) in enumerate(train_pbar):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            reconstructed_inputs, predicted_labels = model(inputs)
            decoder_loss = decoder_criterion(reconstructed_inputs, inputs)
            classifier_loss = classifier_criterion(predicted_labels, labels)

            loss = alpha * decoder_loss + gamma * classifier_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_decoder_loss += decoder_loss.item()
            running_classifier_loss += classifier_loss.item()
            _, predicted = torch.max(predicted_labels.data, 1)
            running_total += labels.size(0)
            running_correct += (predicted == labels).sum().item()
            train_pbar.set_postfix({"loss": running_loss / (i + 1),
                                    "decoder_loss": running_decoder_loss / (i + 1),
                                    "classifier_loss": running_classifier_loss / (i + 1),
                                    "accuracy": 100 * running_correct / running_total})

            optimizer.zero_grad()

        epoch_loss = running_loss / len(train_loader)
        epoch_decoder_loss = running_decoder_loss / len(train_loader)
        epoch_classifier_loss = running_classifier_loss / len(train_loader)
        epoch_train_accuracy = 100 * running_correct / running_total
        train_pbar.set_postfix({"loss": epoch_loss,
                                "decoder_loss": epoch_decoder_loss,
                                "classifier_loss": epoch_classifier_loss,
                                "accuracy": epoch_train_accuracy})

        model.eval()
        test_running_loss = 0.0
        test_running_correct = 0
        test_running_total = 0
        test_pbar = tqdm(test_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (testing)", leave=False)
        with torch.no_grad():
            for inputs, labels in test_pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                reconstructed_inputs, predicted_labels = model(inputs)

                test_metrica.update(predicted_labels, labels)

                decoder_loss = decoder_criterion(reconstructed_inputs, inputs)
                classifier_loss = classifier_criterion(predicted_labels, labels)
                loss = decoder_loss + classifier_loss
                test_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(predicted_labels.data, 1)
                test_running_total += labels.size(0)
                test_running_correct += (predicted == labels).sum().item()
            test_accuracy = 100 * test_running_correct / test_running_total
            test_loss = test_running_loss / len(test_loader.dataset)
        test_pbar.set_postfix({"loss": test_loss,
                               "accuracy": test_accuracy})
        scheduler.step(test_accuracy)
        print(
            'Epoch [{}/{}], Train Loss: {:.4f}, Train Autoencoder Loss: {:.4f}, Train Classification Loss: {:.4f}, Train Accuracy: {:.2f}%, Test Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(
                epoch + 1, num_epochs, epoch_loss, epoch_decoder_loss, epoch_classifier_loss, epoch_train_accuracy,
                test_loss, test_accuracy))

        test_metrica.print_metrics()
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            model_scripted = torch.jit.script(model)
            model_scripted.save("best_model_autoencoder_classifier.pt")

    return best_accuracy, test_metrica


class AECTrainer:
    def __init__(self, df_drug, extraction, Input_size, label_size, num_folds=5, batch_size=64, lr=3e-4, num_epochs=30,
                 index=1):
        self.test_loader = None
        self.df_drug = df_drug
        self.extraction = extraction
        self.Input_size = Input_size
        self.label_size = label_size
        self.num_folds = num_folds
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.best_acc = 0
        self.index = index
        self.metrica = 0

    def train(self):
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        fold = 1
        for train_index, test_index in kf.split(self.extraction):
            if fold == self.index:
                train_extraction = self.extraction.iloc[train_index]
                test_extraction = self.extraction.iloc[test_index]

                train_dataset = DDIDataset(self.df_drug, train_extraction)
                test_dataset = DDIDataset(self.df_drug, test_extraction)
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
                model = AutoencoderClassifier(self.Input_size, self.label_size)
                decoder_criterion = nn.L1Loss()
                classifier_criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,
                                                                       threshold=0.01, cooldown=2)
                self.test_loader = test_loader
                print("Training fold {}...".format(fold))
                self.best_acc, self.metrica = train_autoencoder_classifier(model, train_loader, test_loader
                                                                           , self.label_size,
                                                                           decoder_criterion, classifier_criterion,
                                                                           optimizer, scheduler,
                                                                           self.num_epochs, self.best_acc, 0.5, 1)

            fold += 1


def train_autoencoder(model, train_loader, test_loader, decoder_criterion, optimizer, scheduler, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (training)", leave=False)
        running_loss = 0.0
        running_decoder_loss = 0.0
        for i, (inputs, _) in enumerate(train_pbar):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            reconstructed_inputs = model(inputs)
            decoder_loss = decoder_criterion(reconstructed_inputs, inputs)
            loss = decoder_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_decoder_loss += decoder_loss.item()
            train_pbar.set_postfix({"loss": running_loss / (i + 1), "decoder_loss": running_decoder_loss / (i + 1)})
        epoch_loss = running_loss / len(train_loader)
        epoch_decoder_loss = running_decoder_loss / len(train_loader)
        train_pbar.set_postfix({"loss": epoch_loss, "decoder_loss": epoch_decoder_loss})
        model.eval()
        test_running_loss = 0.0
        test_pbar = tqdm(test_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (testing)", leave=False)
        with torch.no_grad():
            for inputs, _ in test_pbar:
                inputs = inputs.to(device)
                reconstructed_inputs = model(inputs)
                decoder_loss = decoder_criterion(reconstructed_inputs, inputs)
                loss = decoder_loss
                test_running_loss += loss.item() * inputs.size(0)
                test_loss = test_running_loss / len(test_loader.dataset)
                test_pbar.set_postfix({"loss": test_loss})
                scheduler.step(test_loss)
        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Autoencoder Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch + 1,
                                                                                                            num_epochs,
                                                                                                            epoch_loss,
                                                                                                            epoch_decoder_loss,
                                                                                                            test_loss))
    return model


class AutoencoderTrainer:
    def __init__(self, df_drug, extraction, Input_size,num_folds=5, batch_size=64, lr=3e-4, num_epochs=30, index=1):
        self.df_drug = df_drug
        self.extraction = extraction
        self.num_folds = num_folds
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.index = index
        self.test_loader = 0
        self.latent_train = 0
        self.labels_train = 0
        self.model = None
        self.Input_size = Input_size

    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        fold = 1
        for train_index, test_index in kf.split(self.extraction):
            if fold == self.index:
                train_extraction = self.extraction.iloc[train_index]
                test_extraction = self.extraction.iloc[test_index]

                train_dataset = DDIDataset(self.df_drug, train_extraction)
                test_dataset = DDIDataset(self.df_drug, test_extraction)
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

                autoencoder_model = Autoencoder(Input_size=self.Input_size)
                decoder_criterion = nn.L1Loss()
                optimizer = torch.optim.Adam(autoencoder_model.parameters(), lr=self.lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,
                                                                       threshold=0.01, cooldown=2)
                self.test_loader = test_loader
                print("Training fold {}...".format(fold))
                self.model = train_autoencoder(autoencoder_model, train_loader, test_loader, decoder_criterion,
                                               optimizer, scheduler, self.num_epochs)
                latent_train = []
                labels_train = []
                with torch.no_grad():
                    for inputs, labels in train_loader:
                        inputs = inputs.to(device)
                        latent_representations = self.model.encoder(inputs)
                        latent_train.append(latent_representations.cpu().numpy())
                        labels_train.append(labels.numpy())
                    self.latent_train = np.concatenate(latent_train, axis=0)
                    self.labels_train = np.concatenate(labels_train, axis=0)

            fold += 1


