import boto3
import os
import yaml
import pandas as pd
import inspect



# Définition des fonctions pour accéder aux paramètres de configuration
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
input_s3_filepath1 = config['input_s3_filepath1']
output_s3_filepath1 = config['output_s3_filepath1']
input_s3_filepath2 = config['input_s3_filepath2']
output_s3_filepath2 = config['output_s3_filepath2']

s3 = boto3.client("s3",endpoint_url=endpoint_url ,aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, 
                  aws_session_token = aws_session_token)

# Télécharger un fichier depuis un bucket
# Define local paths for downloading 

def get_data():
    local_Raw_data1 = project_root + '/data/first_name.csv'
    local_Raw_data2 = project_root+'/data/transcriptions.csv'
    " le jeu de données firstname_with_sex"
    s3.download_file(bucket_name, input_s3_filepath1, local_Raw_data1)
    "le jeu de données transcriptions_with_sex"
    s3.download_file(bucket_name, input_s3_filepath2, local_Raw_data2)

    data1=pd.read_csv(local_Raw_data1,sep=';')
    data2=pd.read_csv(local_Raw_data2,sep=',')

    print(" Pour le dataset : Firstname", "\n")
    print("shape",data1.shape,"\n")
    print(" Pour le dataset : Transcriptions", "\n")
    print("shape",data2.shape)
    print("\n")

    return data1,data2


