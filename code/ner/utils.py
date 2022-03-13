"""
## Minería de textos
Universidad de Alicante, curso 2021-2022

Esta documentación forma parte de la práctica "[Lectura y documentación de un sistema de
extracción de entidades](https://jaspock.github.io/mtextos/bloque2_practica.html)" y se
basa en el código del curso [CS230](https://github.com/cs230-stanford/cs230-code-examples)
de la Universidad de Stanford.

**Autores de los comentarios:**
    * Laura García Castañeda
    * Diego Silveira Madrid

----------------------- CAMBIARRRRRR -----------------------
Este módulo construye los archivos `words.txt`, `tags.txt` y `dataset_params.json` a partir
del contenido del conjunto de datos del directorio `data/`.

    * El archivo `words.txt` contiene las palabras extraídas de forma única del conjunto de datos
    * El archivo `tags.txt` contiene las etiquetas sintácticas extraídas de forma única del conjunto de datos
    * El archivo `dataset_params.json` contiene los parámetros del conjunto de datos
"""
import json
import logging
import os
import shutil
import torch


# === Clase `Params` ===
class Params():
    """
        Esta clase define una serie de métodos para inicializar, guardar y actualizar la información
        de los parámetros a partir de un archivo JSON.
    """

    # === Método `__init__` ===
    def __init__(self, json_path):
        """
            Este método inicializa los parámetros a partir de los datos de un archivo JSON

            Parámetros:
                * `json_path`: (`str`) ruta del archivo JSON que contiene la información de los parámetros
        """
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    # === Método `save` ===
    def save(self, json_path):
        """
            Este método guarda los parámetros en un archivo JSON

            Parámetros:
                * `json_path`: (`str`) ruta del archivo JSON en el que se guardan los parámetros
        """
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    # === Método `update` ===
    def update(self, json_path):
        """
            Este método actualiza los parámetros a partir de los datos de un archivo JSON

            Parámetros:
                * `json_path`: (`str`) ruta del archivo JSON que contiene la información de los parámetros
        """
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    # === Propiedad `dict` ===
    @property
    def dict(self):
        """
            Esta propiedad contiene la información de los parámetros en formato de diccionario

            Return:
                * `__dict__`: (`dict`) propiedad que contiene los parámetros
        """
        return self.__dict__


# === Clase `RunningAverage` ===
class RunningAverage():
    """
        Esta clase permite calcular el valor medio a partir de las propiedades `steps` y `total`,
        así como también permite inicializar y actualizar el valor de estas propiedades.
    """

    # === Método `__init__` ===
    def __init__(self):
        """
            Este método inicializa los parámetros `steps` y `total` a 0
        """
        self.steps = 0
        self.total = 0

    # === Método `update` ===
    def update(self, val):
        """
            Este método actualiza el valor de la propiedad `total` sumándole el parámetro `val`
            y actualiza el valor de la propiedad `steps` sumándole 1

            Parámetros:
                * `val`: (`int`) Valor que se emplea para actualizar la propiedad `total`
        """
        self.total += val
        self.steps += 1

    # === Método `__call__` ===
    def __call__(self):
        """
            Este método actualiza calcula el valor medio a partir de las propiedades `total`y `steps`
        """
        return self.total / float(self.steps)

# === Función `set_logger` ===
def set_logger(log_path):
    """
        Esta función configura las propiedades que empleará el `logger` para registrar mensajes
        en la terminal y en el archivo `log_path`

        Parámetros:
            * `log_path`: (`str`) ruta del archivo donde se registrarán los mensajes de información
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Configura el formato del mensaje que se escribe en el archivo
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Configura el formato del mensaje que se escribe en la terminal
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


# === Función `save_dict_to_json` ===
def save_dict_to_json(d, json_path):
    """
        Esta función guarda la información contenida en un diccionario en un archivo JSON

        Parámetros:
            * `d`: (`dict`) diccionario con las propiedades del conjunto de datos
            * `json_path`: (`str`) ruta del archivo JSON en el que se guardan los parámetros
    """
    with open(json_path, 'w') as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


# === Función `save_checkpoint` ===
def save_checkpoint(state, is_best, checkpoint):
    """

    :param state:
    :param is_best:
    :param checkpoint:
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


# === Función `load_checkpoint` ===
def load_checkpoint(checkpoint, model, optimizer=None):
   
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint
    