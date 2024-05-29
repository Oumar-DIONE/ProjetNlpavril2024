" Native Package"
import os
import sys
import yaml
import torch

" set filesystem variables"
src_dir=os.path.abspath("../src")
sys.path.append(src_dir)
projet_path = os.path.abspath(os.path.join(src_dir, os.pardir))
sys.path.append(projet_path)
config_path = projet_path+'/config.yaml'
best_model_config_filename="best_model_config.pth"
best_model_filename="best_model.pth"

" self package"
from data import uploade_new_data_from_s3 as load_data
from features import preprocess_Predicted_data as prepocessor
from models import  train_model_Cros_val as cv# Loading the saved model





# Retrieve best model

# Calculer input_len basé sur les données
#input_len = x_train.shape[1]
def upload_model(best_model_config_filename=best_model_config_filename,best_model_filename=best_model_filename,input_len=70):
# Charger la configuration du modèle
    config = torch.load(projet_path+"/models/"+best_model_config_filename)

# Ajouter input_len à la configuration
    config['input_len'] = input_len

# Initialiser le modèle avec les paramètres de la configuration
    loaded_model = cv.classe_model_3_LAYERS(**config)

# Charger les poids du modèle
    loaded_model.load_state_dict(torch.load(projet_path+"/models/"+best_model_filename))

# Mettre le modèle en mode évaluation
    loaded_model.eval()

    print("Model loaded successfully")
    return loaded_model

def predict(file_path_in_s3,file_name_in_local,predicted_data_in_local,best_model_filename,best_model_config_filename):
    #importer les données
    df=load_data.get_data(file_name_in_local,file_path_in_s3,sep=",")
    if "sex" in df.columns:
        input_df=df.drop(["sex"],axis=1)
    else:
        input_df=df
    filename="data_for_prediction"
    # faire le pretraitement des donnes
    preproced_df=prepocessor.preprocess_data(input_df,filename)

    # charger le modéle
    loaded_model=upload_model(best_model_config_filename,best_model_filename,input_len=70)
    # faire la prediction
    with torch.no_grad():  # Désactiver la grad pour les prédictions
        predictions = loaded_model(preproced_df)
        _, predicted = torch.max(predictions, 1)
        predicted_values=predicted.numpy()
        print("Prediction succeful!")

    return predicted_values


