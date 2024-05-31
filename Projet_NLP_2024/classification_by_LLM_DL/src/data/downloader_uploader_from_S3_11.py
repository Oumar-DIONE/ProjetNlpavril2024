import os 
import sys 
import boto3
import yaml

src_dir=os.path.abspath("../src")
project_dir=os.path.abspath(os.path.join(os.path.dirname(src_dir), '.')) 
sys.path.append(project_dir)
config_path=project_dir+"/config1.yaml"
config_path
with open(config_path,'r') as f:
    config=yaml.safe_load(f)

aws_access_key_id=config['aws_access_key_id']
aws_secret_access_key=config['aws_secret_access_key']
endpoint_url=config['endpoint_url']
aws_session_token = config['aws_session_token']
bucket_name = config['bucket_name']




def get_bucket_lists(aws_access_key_id=aws_access_key_id,aws_secret_access_key=aws_secret_access_key,endpoint_url=endpoint_url,aws_session_token = aws_session_token):
    # Initialisez le client S3
    s3 = boto3.client('s3',aws_access_key_id=aws_access_key_id,aws_secret_access_key=aws_secret_access_key,endpoint_url=endpoint_url,aws_session_token = aws_session_token)  # Uniquement pour MinIO)
    # Obtenir la liste des buckets
    response = s3.list_buckets()

    # Afficher les noms des buckets
    print("Liste des buckets S3 :")
    for bucket in response['Buckets']:
        print(f"- {bucket['Name']}")

def get_object_lists(bucket_name=bucket_name ,aws_access_key_id=aws_access_key_id,aws_secret_access_key=aws_secret_access_key,endpoint_url=endpoint_url,aws_session_token = aws_session_token):
    # Initialisez le client S3
    s3 = boto3.client('s3',aws_access_key_id=aws_access_key_id,aws_secret_access_key=aws_secret_access_key,endpoint_url=endpoint_url,aws_session_token = aws_session_token)  # Uniquement pour MinIO)
    # Obtenir la liste des objets dans le bucket
    response = s3.list_objects_v2(Bucket=bucket_name)

# Afficher la liste des objets
    print("Liste des objets dans le bucket", bucket_name, ":")
    for obj in response.get('Contents', []):
         print("- ", obj['Key'])

" Initialiser un client S3 pour le téchargement depuis et l'envoie ver S3"
s3 = boto3.client('s3',aws_access_key_id=aws_access_key_id,aws_secret_access_key=aws_secret_access_key,endpoint_url=endpoint_url,aws_session_token = aws_session_token)  # Uniquement pour MinIO)


def download_from_S3(bucket_name=bucket_name, s3_file_path="Socface_Data/data_for_prediction.csv", local_file_path=project_dir+"/data/local_data.csv"):
    # Télécharger le fichier depuis le bucket
    s3.download_file(bucket_name,s3_file_path , local_file_path)

    print("Le fichier a été téléchargé avec succès.")


def upload_to_S3(bucket_name=bucket_name, s3_file_path="Socface_Data/data_for_prediction_uploaded.csv", local_file_path=project_dir+"/data/local_data.csv"):
    # Télécharger le fichier depuis le bucket
    s3.upload_file(local_file_path, bucket_name, s3_file_path)

    print("Le fichier a été envoyé (vers s3) avec succès.")

upload_to_S3()
