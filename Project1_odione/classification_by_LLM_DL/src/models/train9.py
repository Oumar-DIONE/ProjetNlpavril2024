" import native Packages"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset  
import pandas as pd
import os
import sys

" set filesystem variables"
src_dir = os.path.abspath('../src')
sys.path.append(src_dir)

" import self packages"

from features import make_fetaure20


# Définir l'architecture du réseau de neurones
class classe_model_3_LAYERS(nn.Module):
    "Cette méthode __init__ est le constructeur de la classe et est utilisée pour définir les couches et les paramètres du modèle."
    def __init__(self,l1,l2,l3):
        # Convertir les données en tenseurs PyTorch
        (x_train,y_train),(x_test,y_test)=make_fetaure20.preprocess_data()
        self.tenseurs = torch.stack(x_train.tolist())
        self.labels = torch.tensor(y_train.tolist())
        self.tenseurs_test = torch.stack(x_test.tolist())
        self.labels_test = torch.tensor(y_test.tolist())
        self.input_len=max(len(self.tenseurs[i]) for i in range(len(self.tenseurs)))
        super(classe_model_3_LAYERS, self).__init__()
        self.l1=l1
        self.l2=l2
        self.l3=l3
        self.couche_lineaire_1 = nn.Linear(self.input_len, l1)  # Couche linéaire avec input_len entrées et l1 sorties
        self.couche_lineaire_2 = nn.Linear(l1, l2)  # Couche linéaire avec l1 entrées et l2 sorties
        self.couche_lineaire_3 = nn.Linear(l2, l3)  # Couche linéaire avec l2 entrées et l1 sorties
        self.couche_lineaire_4 = nn.Linear(l3, 2)   # Couche linéaire avec l3 entrées et 2 sorties (2 classes)
    "Cette forward définit comment les données d'entrée traversent les couches définies pour produire une sortie."

    def forward(self, x):
        x = torch.relu(self.couche_lineaire_1(x.float()))  # Convertir les tenseurs d'entrée en type torch.float
        x = self.couche_lineaire_2(x)
        x = self.couche_lineaire_3(x)

        x = self.couche_lineaire_4(x)             # Pas de fonction d'activation pour la dernière couche
        return x
    # Créer un DataLoader pour les données d'entraînement
    def load_data(self,batch_size=12):
        dataset = TensorDataset(self.tenseurs,self.labels)
        train_loader = DataLoader(dataset, batch_size)

        return train_loader


def training(l1,l2,l3,lr=0.001,num_epochs=100):
        # Initialiser le modèle
        model = classe_model_3_LAYERS(l1=l1,l2=l2,l3=l3)
        train_loader=model.load_data()


        # Définir la fonction de perte et l'optimiseur
        criterion = nn.CrossEntropyLoss()  # Fonction de perte pour les problèmes de classification
        optimizer = optim.Adam(model.parameters(), lr=lr)  # Optimiseur Adam avec un taux d'apprentissage de 0.001
        # Entraîner le modèle
        # Variables pour le calcul de l'exactitude
        total_correct = 0
        total_samples = 0

        # Initialiser l'exactitude précédente à zéro
        prev_accuracy = 0.0

# Entraîner le modèle
        for epoch in range(num_epochs):
            for batch_tenseurs, batch_labels in train_loader:
                optimizer.zero_grad()
                predictions = model(batch_tenseurs)
                loss = criterion(predictions, batch_labels)
                loss.backward()
                optimizer.step()

        # Calculer les prédictions correctes
                _, predicted = torch.max(predictions, 1)
                total_correct += (predicted == batch_labels).sum().item()
                total_samples += batch_labels.size(0)

    # Calculer l'exactitude à la fin de chaque époque
            accuracy = total_correct / total_samples
    #print(f'Epoch [{epoch+1}/{num_epochs}], Accuracy: {accuracy}')

    # Vérifier si l'exactitude diminue
            if accuracy < prev_accuracy:
                print("Accuracy decreased. Stopping training.")
                print(f'Epoch [{epoch+1}/{num_epochs}], Accuracy: {prev_accuracy}')

                break
            elif epoch+1==num_epochs:
                print(f'Epoch [{epoch+1}/{num_epochs}], Accuracy: {accuracy}')
    
            # Mettre à jour l'exactitude précédente
            prev_accuracy = accuracy
    #performance1=pd.DataFrame({'Model_name':"model1","data":"Train","Accuracy":prev_accuracy})
            performance1=pd.DataFrame(["model1","Train",prev_accuracy])

            
        return performance1