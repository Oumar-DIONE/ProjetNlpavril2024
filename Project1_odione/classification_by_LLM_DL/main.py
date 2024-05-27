from dotenv import load_dotenv
import os
import sys

# Charger les variables d'environnement à partir du fichier .env
load_dotenv()

# Ajouter le dossier 'src' au PYTHONPATH
sys.path.append(os.getenv('PYTHONPATH'))

# Maintenant, vous pouvez importer vos modules
import src.data.make_dataset as make_data

print("hello world")