import os
import yaml


# Définition des fonctions pour accéder aux paramètres de configuration
def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config


# Obtenir le chemin absolu du répertoire racine du projet
import inspect
import os

# Obtenir le chemin absolu du module
module_path = inspect.getfile(inspect.currentframe())
module_dir = os.path.dirname(module_path)
project_root =  os.path.abspath(os.path.join(os.path.dirname(module_dir), '..')) 
config_path = project_root+"/config.yaml"





print(config_path)
