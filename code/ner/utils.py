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

Este módulo contiene una serie de clases y funciones que se utilizarán como *helpers* a lo largo del proyecto.

    * La clase `Params` se utilizará para guardar y poder acceder a las propiedades del dataset con el que se
    está trabajando.

    * La clase `runningAverage` se utilizará para mantener actualizada la media de un valor.

    * La función `set_logger()` se utilizará para configurar los mensajes que aparecerán en el *log*.

    * La función `save_dict_to_json()` se utilizará para guardar la información de una determinada variable
    tipo diccionario en un archivo JSON.

    * La función `save_checkpoint()` se utilizará para guardar en un archivo la situación de un modelo en
    determinadas iteraciones del entrenamiento y, además, guardar en otro archivo la que mejores resultados
    proporciona.

    * La función `load_checkpoint()` se utiliza para cargar la información de un modelo guardado previamente
    en un archivo.
"""

# === Importar las librerías ===
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
            Este método inicializa los parámetros a partir de los datos de un archivo JSON.

            **Parámetros:** `json_path` (`str`): ruta del archivo JSON que contiene la información de los parámetros.
        """
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    # === Método `save` ===
    def save(self, json_path):
        """
            Este método guarda los parámetros en un archivo JSON.

            **Parámetros:** `json_path`: (`str`) ruta del archivo JSON en el que se guardan los parámetros.
        """
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    # === Método `update` ===
    def update(self, json_path):
        """
            Este método actualiza los parámetros a partir de los datos de un archivo JSON.

            **Parámetros:** `json_path`: (`str`) ruta del archivo JSON que contiene la información de los parámetros.
        """
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    # === Propiedad `dict` ===
    @property
    def dict(self):
        """
            Esta propiedad contiene la información de los parámetros en formato de diccionario.

            **Return:** `__dict__`: (`dict`) propiedad que contiene los parámetros.
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
            Este método inicializa los parámetros `steps` y `total` a 0.
        """
        self.steps = 0
        self.total = 0

    # === Método `update` ===
    def update(self, val):
        """
            Este método actualiza el valor de la propiedad `total` sumándole el parámetro `val`
            y actualiza el valor de la propiedad `steps` sumándole 1.

            **Parámetros:** `val`: (`int`) valor que se emplea para actualizar la propiedad `total`.
        """
        self.total += val
        self.steps += 1

    # === Método `__call__` ===
    def __call__(self):
        """
            Este método calcula el valor medio a partir de las propiedades `total`y `steps`.
        """
        return self.total / float(self.steps)


# === Función `set_logger` ===
def set_logger(log_path):
    """
        Esta función configura las propiedades que empleará el `logger` para registrar mensajes
        en la terminal y en el archivo `log_path`.

        **Parámetros:** `log_path`: (`str`) ruta del archivo donde se registrarán los mensajes de información.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Configura el formato del mensaje que se escribe en el archivo.
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Configura el formato del mensaje que se escribe en la terminal.
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


# === Función `save_dict_to_json` ===
def save_dict_to_json(d, json_path):
    """
        Esta función guarda la información contenida en un diccionario en un archivo JSON.

        **Parámetros:**

        - `d`: (`dict`) diccionario con las propiedades del conjunto de datos.

        - `json_path`: (`str`) ruta del archivo JSON en el que se guardan los parámetros.
    """
    with open(json_path, 'w') as f:
        # Iteramos por cada uno de los elementos del diccionario para convertirlos a *float*.
        # Ya que JSON no soporta las variables de *NumPy*.
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


# === Función `save_checkpoint` ===
def save_checkpoint(state, is_best, checkpoint):
    """
        Esta función guarda los parámetros del modelo en un determinado momento del entrenamiento. Además, si este
        es el mejor modelo encontrado hasta el momento, lo guarda en una dirección adicional.

        **Parámetros:**

        - `state`: (`dict`) contiene el parámetro `state_dict` del modelo, el cuál contiene toda la información
        necesaria para replicar el modelo.

        - `ìs_best`: (`bool`) variable que indica si este modelo es el mejor encontrado hasta el momento.

        - `checkpoint`: (`str`) directorio en el que se guardará la información del *checkpoint*.
    """

    # Si todavía no existe el directorio en el que se guardan los *checkpoints*, se crea.
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")

    # Guarda el modelo encontrado en el directorio indicado.
    torch.save(state, filepath)

    # Si el modelo es el mejor que se ha encontrado hasta el momento, el modelo es guardado también en el directorio
    # reservado para el mejor modelo.
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


# === Función `load_checkpoint` ===
def load_checkpoint(checkpoint, model, optimizer=None):
    """
        Esta función utiliza la información guardada de un modelo en un determinado momento para cargarla de nuevo y
        poder trabajar con ella al instante sin necesidad de realizar un nuevo entrenamiento.

        **Parámetros:**

        - `checkpoint`: (`str`) directorio en el se encuentra guardada la información del modelo.

        - `model`: (`torch.nn.Module`) modelo en el que vamos a cargar los datos guardados en el *checkpoint*.
            Este debe tener la misma estructura que el que se utilizó para guardar el modelo.

        - `optimizer`: (`torch.optim`) estructura donde se guardará el optimizador que utliza el modelo en caso de
            que este fuera guardado en el *checkpoint* del modelo.
    """

    # Si el archivo de *checkpoint* no existe, se lanza una excepción indicando que el archivo no existe.
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))

    # Se carga el modelo en forma de diccionario.
    checkpoint = torch.load(checkpoint)
    # Se crea el modelo a partir de la información del diccionario.
    model.load_state_dict(checkpoint['state_dict'])

    # En el caso de que se haya guardado el optimizador, este también se carga.
    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint
    