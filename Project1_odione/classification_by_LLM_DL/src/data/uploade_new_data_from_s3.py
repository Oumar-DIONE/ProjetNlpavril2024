import boto3
import os
import yaml
import pandas as pd
import inspect

" Définition des fonctions pour accéder aux paramètres de configuration"


def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config


# Obtenir le chemin absolu du répertoire racine du projet
module_path = inspect.getfile(inspect.currentframe())
module_dir = os.path.dirname(module_path)
project_root =  os.path.abspath(os.path.join(os.path.dirname(module_dir), '..')) 


# Charger la configuration
config_path = project_root+'/config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

aws_access_key_id = config['aws_access_key_id']
aws_secret_access_key = config['aws_secret_access_key']
aws_session_token = config['aws_session_token']
endpoint_url = config['endpoint_url']
bucket_name = config['bucket_name']


s3 = boto3.client("s3",endpoint_url=endpoint_url ,aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, aws_session_token = aws_session_token)

# Télécharger un fichier depuis un bucket
# Define local paths for downloading 

def get_data(file_name_in_local,file_path_in_s3,sep):
    local_Raw_df = project_root + '/data/'+file_name_in_local
    s3.download_file(bucket_name, file_path_in_s3, local_Raw_df)
    df=pd.read_csv(local_Raw_df,sep=sep)

    print(" Pour le dataset : ",file_name_in_local, "\n")
    print("shape",df.shape,"\n")

    return df


