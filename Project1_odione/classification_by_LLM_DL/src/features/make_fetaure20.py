# Important native module
import warnings
warnings.filterwarnings("ignore")
import numpy as np 
import pandas as pd 
import random as rd
import torch
from transformers import BertTokenizer, BertModel
import inspect
import os
import sys
from unidecode import unidecode
import re
import difflib
import boto3
from botocore.client import Config
from sklearn.model_selection import train_test_split
import yaml




tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# set constant
# fixer les graines de générateurs de nombres aléatoires afin d'avoir des résultats reproductibles
seed=1996
np.random.seed(seed)
rd.seed(seed)


# filesysteme variable
src_dir = os.path.abspath('../src')
sys.path.append(src_dir)
file_path = inspect.getfile(inspect.currentframe())
file_dir = os.path.dirname(file_path)
project_root =  os.path.abspath(os.path.join(os.path.dirname(file_dir), '..')) 
firstname="first_name.csv"
transcriptions="transcriptions.csv"
# modéle
n_max_token=12
# Important self module
from data import make_dataset3
#from data import make_dataset3

" définition de l'ensemble des fonctions intermédiaires nécessaires pour faire le preprcessing"
" Convertir en minuscules et supprimer les accents"


def str_lower(string):
     chaine_traitee = unidecode(string.lower())
     return chaine_traitee
# Caractères spéciaux à supprimer
special_chars = ['"', "''", '°', ';', '\\', '(', ')']

# Définir une fonction pour supprimer les caractères spéciaux
def remove_special_chars(chaine):
    for char in special_chars:
        chaine = chaine.replace(char, '')
    return chaine


def ajouter_virgule_avant_mots_specifiques(phrase):
    # Liste des mots spécifiques à trouver
    mots_specifiques = ['prenom', 'date_naissance', 'lieux_naissance', 'relation', 'employeur',"profession"]
    # Expression régulière pour trouver les mots spécifiques suivis de deux-points
    regex = r'(' + '|'.join(mots_specifiques) + r')\s*:'
    # Remplacez chaque occurrence de la regex par une virgule suivie du mot spécifique
    phrase_propre = re.sub(regex, r',\1:', phrase)
    return phrase_propre

# supprimer les espace avant et aprés les :
def nettoyer_phrase(phrase):
    # Séparez la phrase en couples "mot1:mot2" en utilisant la virgule comme délimiteur
    couples = phrase.split(',')
    # Parcourez chaque couple
    couples_nettoyes = []
    for couple in couples:
        # Supprimez les espaces avant et après les deux points
        couple_nettoye = couple.replace(' :', ':').replace(': ', ':')
        couples_nettoyes.append(couple_nettoye)
    # Rejoignez les couples pour former une nouvelle phrase
    phrase_nettoyee = ', '.join(couples_nettoyes)
    return phrase_nettoyee

# Définir la fonction pour récupérer et remplacer le nom
def recuperer_et_remplacer_nom(phrase, nouveau_nom):
    # Séparer la phrase en couples "mot1:mot2" en utilisant la virgule comme délimiteur
    couples = phrase.split(',')
    # Parcourir chaque couple
    for i, couple in enumerate(couples):
        # Supprimer les espaces avant et après les deux points
        couple_nettoye = couple.replace(' :', ':').replace(': ', ':')
        # Séparer le couple en mot1 et mot2 en utilisant les deux points comme délimiteur
        mots = couple_nettoye.split(':')
        # Si le mot1 est "nom", remplacer le mot2 par le nouveau nom
        if mots[0].strip() == 'nom':
            couples[i] = 'nom:' + nouveau_nom
    # Rejoindre les couples pour former une nouvelle phrase
    phrase_modifiee = ','.join(couples)
    return phrase_modifiee

# Définir la fonction pour récupérer le nom
def recuperer_nom(phrase,feature='nom'):
    # Séparer la phrase en couples "mot1:mot2" en utilisant la virgule comme délimiteur
    couples = phrase.split(',')
    # Parcourir chaque couple
    for couple in couples:
        # Supprimer les espaces avant et après les deux points
        couple_nettoye = couple.replace(' :', ':').replace(': ', ':')
        # Séparer le couple en mot1 et mot2 en utilisant les deux points comme délimiteur
        mots = couple_nettoye.split(':')
        # Si le mot1 est "nom", retourner le mot2
        if mots[0].strip() == feature:
            return mots[1].strip()
    # Si aucun nom n'est trouvé, retourner une chaîne vide
    return ''
def corriger_noms(liste_1, liste_2):
    corrections = {}
    for nom_1 in liste_1:
        # Calculer la similarité avec chaque nom de la deuxième liste
        ratios = [difflib.SequenceMatcher(None, nom_1, nom_2).ratio() for nom_2 in liste_2]
        # Trouver le nom de la deuxième liste avec la plus grande similarité
        index_max_ratio = ratios.index(max(ratios))
        nom_correct = liste_2[index_max_ratio]
        # Ajouter la correction à un dictionnaire
        corrections[nom_1] = nom_correct
    
    # Remplacer les noms dans liste_1 par les noms corrigés
    liste_1_corrigee = [corrections.get(nom, nom) for nom in liste_1]
    
    return liste_1_corrigee

def get_features(phrases):
    vect=phrases.split(',')
    features=[texte.split(":")[0] for texte in vect]
    return features
def get_overall_feats(df):
    feats=[]
    for line_ in df.prediction:
        feats=feats +get_features(line_)
    return set(feats)
def Binarizer(df,col_to_bin="sex"):
    # Rendre Binaire la variable sex : homme=1
    df["Sex_num"]=1
    df["Sex_num"][df[col_to_bin]=="femme"]=0
    df=df.drop([col_to_bin],axis=1)
    return df

def bert_preprocessing(df_w):
# Instanciez le tokenizer de BERT
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Initialisation des listes pour stocker les input_ids et les prénoms
    input_ids_list = []
    prenoms_list = []
    labels=df_w.labels.values

# Trouver la longueur maximale des input_ids
    max_length = 0
    tes_Str_input=list(df_w.Input_text.values)
# Boucle sur chaque phrase de la liste
    for phrase in tes_Str_input:
    # Tokenisez la phrase avec le tokenizer de BERT
        encoding = tokenizer(phrase, return_tensors='pt')
    
    # Récupérez le prénom dans la phrase
        if len(phrase.split(", ,"))>1:
            prenom = phrase.split(", ,")[1].split(":")[1].strip()  # Extraction du prénom
            prenoms_list.append(prenom)
        else:
            prenom = phrase.split(",")[1].split(":")[1].strip()  # Extraction du prénom
            prenoms_list.append(prenom)

    # Récupérez les input_ids
        input_ids = encoding['input_ids'].squeeze()  # squeeze() pour obtenir un tenseur à une dimension
        input_ids_list.append(input_ids)

    # Mettre à jour la longueur maximale des input_ids
        max_length = max(max_length, len(input_ids))

# Appliquer le padding sur les input_ids pour qu'ils aient tous la même longueur
    padded_input_ids_list = [torch.cat([input_ids, torch.zeros(max_length - len(input_ids), dtype=torch.long)]) for input_ids in input_ids_list]

# Créez un DataFrame à partir des listes
    Final_input = pd.DataFrame({'Phrase': tes_Str_input, 'prenom': prenoms_list, 'input_ids': padded_input_ids_list,"labels":labels})

# Affichez le DataFrame
    return Final_input

def preprocess_data():
    FN_df,Trs_df=make_dataset3.get_data()
    " preprocessing du dataframe firstname "
    FN_df.firstname=FN_df.firstname.apply(lambda x: str_lower(x))
    FN_df["Male_name_fred"]=FN_df["male"]/(FN_df["male"]+FN_df["female"])
    # recupérer et stock les noms et frequences associées
    Probas=FN_df["Male_name_fred"].values
    Names=FN_df["firstname"].values
    " preprocessing du dataframe firstname "
    columns=Trs_df.columns
    Trs_df=Trs_df.drop(['subject_line'],axis=1)
    Trs_df.prediction=Trs_df.prediction.apply(lambda x: str_lower(x))
    Trs_df = Trs_df.applymap(remove_special_chars)
    # Ne gardez que les colonnes intéressantes pour l'étude
    df_w=Trs_df.copy()
    cols_to_drop=[ 'groundtruth']
    df_w=df_w.drop(cols_to_drop,axis=1)
    df_w["prediction"]=df_w.prediction.apply(lambda x : nettoyer_phrase(x))
    df_w["prediction"]=df_w.prediction.apply(lambda x : ajouter_virgule_avant_mots_specifiques(x))
    df_w["prediction"]=df_w.prediction.apply(lambda x :  recuperer_et_remplacer_nom(x, recuperer_nom(x)))
    " je préfére faire une suppression par valeur où à la limite garder plutot que de supprimer"
    features=get_overall_feats(df_w)
    features=list(features)
    k=len(features)
    index_=list(set(range(k)).difference(set([2,3])))
    feats=[features[i] for i in index_]
    df_w=Binarizer(df_w)
    df_w=df_w.rename(columns={"prediction":"Input_text","Sex_num":"labels"})
    df_w.to_csv(project_root+"/data/Clean_Data_before_tokenized.csv")
    # Transnformer les données en lists
    texte_liste=df_w.Input_text.values
    texte_liste=list(texte_liste)
    Final_input=bert_preprocessing(df_w)
    #ajout du colonne des inputs
    Final_input["labels"]=df_w['labels'].values
    Final_input=Final_input.drop(["Phrase"],axis=1)
    Final_input=Final_input.set_index("prenom")
    Train_data,Test_data=train_test_split(Final_input,test_size=0.2)
    X_train,Y_train=Train_data.input_ids,Train_data.labels
    X_test,Y_test=Test_data.input_ids,Test_data.labels
    print( " Train :", X_train.shape,Y_train.shape)
    print( " Test :", X_test.shape,Y_test.shape)
    Final_input.to_csv(project_root+"/data/df_after_tokenizeation.csv")
    print("well done")
    return  (X_train,Y_train),(X_test,Y_test)

config_path=project_root+"/config.yaml"
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
df_after_tokenizeation_in_s3 = config['df_after_tokenizeation']

file_path=project_root+"/data/df_after_tokenizeation.csv"

def upload_preproces_in_online_Bucket(file_path=file_path,bucket_name=bucket_name,object_name=df_after_tokenizeation_in_s3):
        # Initialisation du client boto3 pour MinIO
    s3_client = boto3.client(
    's3',
    endpoint_url=endpoint_url,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    config=Config(signature_version='s3v4')
)   

# Envoi du fichier
    try:
        s3_client.upload_file(file_path, bucket_name, object_name)
        print(f"Fichier '{file_path}' envoyé avec succès au bucket '{bucket_name}' sous le nom '{object_name}'.")
    except Exception as e:
         print(f"Une erreur s'est produite lors de l'envoi du fichier : {e}")