" import native Packages"
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset  
import pandas as pd
import os
import sys
from torchsummary import summary  # Importer la fonction summary de torchsummary
import numpy as np
from sklearn.model_selection import GridSearchCV


" set filesystem variables"
src_dir = os.path.abspath('../src')
sys.path.append(src_dir)

" import self packages"

from features import make_fetaure20

# Définir l'architecture du réseau de neurones
class classe_model_3_LAYERS(nn.Module):
    "Cette méthode __init__ est le constructeur de la classe et est utilisée pour définir les couches et les paramètres du modèle."
    def __init__(self,l1,l2,l3,input_len=70):
        super(classe_model_3_LAYERS, self).__init__()
        self.l1=l1
        self.l2=l2
        self.l3=l3
        self.input_len=input_len
        self.couche_lineaire_1 = nn.Linear(self.input_len, l1)  # Couche linéaire avec input_len entrées et l1 sorties
        self.couche_lineaire_2 = nn.Linear(l1, l2)  # Couche linéaire avec l1 entrées et l2 sorties
        self.couche_lineaire_3 = nn.Linear(l2, l3)  # Couche linéaire avec l2 entrées et l1 sorties
        self.couche_lineaire_4 = nn.Linear(l3, 2)   # Couche linéaire avec l3 entrées et 2 sorties (2 classes)
    "Cette forward définit comment les données d'entrée traversent les couches définies pour produire une sortie."

    def forward(self, x):
        # Convertir la série en un tableau numpy contenant les tenseurs
        x = np.array(x.tolist())  # Convertir la série en une liste, puis en un tableau numpy
        # Convertir le tableau numpy en un tenseur PyTorch
        x = torch.tensor(x).float()  # Convertir X en tenseur PyTorc
        #x = torch.relu(self.couche_lineaire_1(x.float()))  # Convertir les tenseurs d'entrée en type torch.float
        x=torch.relu(self.couche_lineaire_1(x))
        x = self.couche_lineaire_2(x)
        x = self.couche_lineaire_3(x)

        x = self.couche_lineaire_4(x)             # Pas de fonction d'activation pour la dernière couche
        return x
    # Créer un DataLoader pour les données d'entraînement
    def load_data(self,X,Y,batch_size=10):
        tenseurs = torch.stack(X.tolist())
        labels = torch.tensor(Y.tolist())
        dataset = TensorDataset(tenseurs,labels)
        train_loader = DataLoader(dataset, batch_size)
        return train_loader

class PyTorchModel(BaseEstimator, ClassifierMixin):
    def __init__(self, l1=128, l2=64, l3=32, lr=0.001, batch_size=12, num_epochs=100):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model = None
    def fit(self,X,Y):
        # Initialiser le modèle
        self.model = classe_model_3_LAYERS(l1=self.l1,l2=self.l2,l3=self.l3)
        train_loader =self.model.load_data(X,Y,self.batch_size)
        # Définir la fonction de perte et l'optimiseur
        criterion = nn.CrossEntropyLoss()  # Fonction de perte pour les problèmes de classification
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)  # Optimiseur Adam avec un taux d'apprentissage de 0.001
        # Entraîner le modèle

# Entraîner le modèle
        for epoch in range(self.num_epochs):
            for batch_tenseurs, batch_labels in train_loader:
                optimizer.zero_grad()
                predictions = self.model(batch_tenseurs)
                loss = criterion(predictions, batch_labels)
                loss.backward()
                optimizer.step()
        return self
    def predict(self,X):
            self.model.eval()
            # Convertir la série en un tableau numpy contenant les tenseurs
            array_X = np.array(X.tolist())  # Convertir la série en une liste, puis en un tableau numpy
            # Convertir le tableau numpy en un tenseur PyTorch
            tensor_X = torch.tensor(array_X).float()  # Convertir X en tenseur PyTorch de type float
    
    
            with torch.no_grad():
                predictions = self.model(X)
                _, predicted = torch.max(predictions, 1)
            return predicted.numpy()

    def score(self,X,Y):
            y_pred=self.predict(X)
            total_correct=(y_pred == Y).sum()
            total_samples=len(Y)
            accurracy=total_correct/total_samples
            return accurracy


    
class make_grid_search:
    def __init__(self,X,Y,grid_params,n_cv=10):
        self.grid_params=grid_params
        self.n_cv=n_cv
        self.X=X
        self.Y=Y
        self.best_params=None
        self.best_model=None
        self.best_score=None
        # Create our Pytorch  classifier
        classifier = PyTorchModel()
        # Perform grid search cross-validation
        grid_search = GridSearchCV(estimator=classifier,param_grid=self.grid_params,cv=n_cv)
        grid_search.fit(self.X, self.Y)
        self.best_params=grid_search.best_params_
        self.best_model = grid_search.best_estimator_
        self.best_score=grid_search.best_score_
    def show_results(self):
        # Print the best hyperparameters and the corresponding mean cross-validated score
        print("Best Hyperparameters: ", self.best_params,"\n")
        print("Best Score: ", self.best_score)
    



