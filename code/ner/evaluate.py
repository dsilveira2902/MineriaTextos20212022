"""
## Minería de textos
Universidad de Alicante, curso 2021-2022

Esta documentación forma parte de la práctica "[Lectura y documentación de un sistema de
extracción de entidades](https://jaspock.github.io/mtextos/bloque2_practica.html)" y se
basa en el código del curso [CS230](https://github.com/cs230-stanford/cs230-code-examples)
de la Universidad de Stanford.

*Autores de los comentarios:*

    * Laura García Castañeda
    * Diego Silveira Madrid

En este módulo se define la evaluación de los modelos. Para ello, en primer lugar, se cargan los
datos de testeo. A continuación, se carga un modelo a partir de su información guardada previamente.
Seguidamente, se realiza la evaluación del modelo mediante la precisión y la función de pérdida y,
finalmente, se guardan los resultados obtenidos en un archivo JSON.
"""

# === Importar las librerías ===
import argparse
import logging
import os

import numpy as np
import torch
import utils
import model.net as net
from model.data_loader import DataLoader

"""
Definimos los argumentos que recibirá nuestro archivo por línea de comandos.
"""
# Creamos el objeto que maneja los argumentos gracias a la librería `argparse`.
parser = argparse.ArgumentParser()
# Parámetro para definir el directorio que contiene el dataset. Por defecto será `data/small`.
parser.add_argument('--data_dir', default='data/small', help="Directory containing the dataset")
# Parámetro para definir el directorio que contiene la configuración del modelo.
# Por defecto será `experiments/base_model`.
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
# Parámetro para definir el archivo que contiene el modelo a cargar.
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


# === Método `evaluate` ===
def evaluate(model, loss_fn, data_iterator, metrics, params, num_steps):
    """
        Evalúa el modelo en `num_steps` batches.

        Parámetros:

        - `model`: (`torch.nn.Module`) es un objeto de la clase Net, que hace referencia a la red neuronal.

        - `loss_fn`: ( `function` ) función de pérdida a utilizar para evaluar el modelo en cada uno de los batches.

        - data_iterator: (`generator`) generador que produce *batches* de datos y etiquetas.

        - metrics: (`dict`) diccionario que contiene como clave el nombre de la métrica y como valor la función
          que se usa para calcularla.

        - params: (`Params`) objeto de la clase Params que contiene las propiedades del conjunto de datos.

        - num_steps: (`int`) número de *batches* a ejecutar en el entrenamiento.

    Return:

        - `metrics_mean`: (`dict`) diccionario que contiene el valor medio para cada una de las métricas en
          todos los *batches*.
    """

    # Establece el modelo en modo de evaluación.
    model.eval()

    # Define una lista que contiene, para cada uno de los batches, un diccionario
    # con el resultado de la función de pérdida y de cada una de las métricas
    # calculadas.
    summ = []

    # Itera sobre cada uno de los *batches*
    for _ in range(num_steps):

        # Obtiene el siguiente *batch* de datos y de labels
        data_batch, labels_batch = next(data_iterator)

        # Dado un batch de datos, el modelo calcula la salida
        output_batch = model(data_batch)
        # Se calcula la función de pérdida a partir de la salida esperada
        # y la salida predicha por el modelo.
        loss = loss_fn(output_batch, labels_batch)

        # Toma la salida predicha por el modelo, la mueve a la CPU, y la convierte en un array de NumPy
        output_batch = output_batch.data.cpu().numpy()
        # Toma la salida esperada, la mueve a la CPU, y la convierte en un array de NumPy
        labels_batch = labels_batch.data.cpu().numpy()

        # Crea un diccionario con los valores de cada una de las métricas para ese *batch*,
        # en el cual, las claves son el nombre de la métrica, y el valor es el resultado
        # de aplicar la función de la métrica.
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}

        # Añade el valor de pérdida al diccionario
        summary_batch['loss'] = loss.item()

        # Añade el diccionario de métricas de un batch determinado a la lista
        summ.append(summary_batch)

    # Crea un diccionario con el valor medio de cada de las métricas de todos los *batches*
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}

    # A partir del diccionario `metrics_mean` genera un string con el valor medio de cada una
    # de las métricas de todos los *batches* separadas por punto y coma.
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())

    # Muestra un mensaje de información en el *logger* con el resultado de las métricas.
    logging.info("- Eval metrics : " + metrics_string)

    # Devuelve el diccionario con el valor medio de cada de las métricas
    return metrics_mean


# === Ejecución principal del archivo ===
if __name__ == '__main__':

    # Argumentos de entrada
    args = parser.parse_args()

    # Ruta del archivo que contiene la configuración del modelo.
    json_path = os.path.join(args.model_dir, 'params.json')
    # Se comprueba que el archivo se encuentra en la ruta indicada.
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    # Se cargan los valores de la configuración del modelo como un objeto de la clase `Params`
    params = utils.Params(json_path)

    # Indica si el uso de la GPU esta disponible
    params.cuda = torch.cuda.is_available()

    # Establece un valor semilla de manera *pseudo-aleatoria* para poder reproducir resultados experimentales
    # con los modelos
    torch.manual_seed(230)

    if params.cuda:
        torch.cuda.manual_seed(230)

    # Crea el archivo de *logs* en la ruta indicada como parámetro
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Muestra un mensaje de información en el archivo de *logs* indicando el comienzo de la carga de los datos.
    logging.info("Creating the dataset...")

    # Crea un objeto de la clase `DataLoader` en el cual se guarda la información relativa al dataset
    data_loader = DataLoader(args.data_dir, params)

    # Carga los datos del directorio 'test'
    data = data_loader.load_data(['test'], args.data_dir)

    # Guarda los datos de testeo seleccionados en una variable
    test_data = data['test']

    # Guarda el tamaño de los datos de testeo seleccionados como una propiedad del conjunto de datos
    params.test_size = test_data['size']

    # Crea un generador que produce *batches* de datos y etiquetas a partir de los datos de testeo
    test_data_iterator = data_loader.data_iterator(test_data, params)

    # Muestra un mensaje de información en el archivo de *logs* indicando el fin de la carga de los datos.
    logging.info("- done.")

    # Comprueba si la GPU está disponible. En caso afirmativo, se crea una copia del modelo en la GPU
    # y, en caso de que no se encuentre disponible, crea el modelo en la CPU.
    model = net.Net(params).cuda() if params.cuda else net.Net(params)

    # Obtiene la función de pérdida del modelo
    loss_fn = net.loss_fn
    # Obtiene un diccionario con las métricas del modelo
    metrics = net.metrics

    # Muestra un mensaje de información en el archivo de *logs* indicando el inicio de la evaluación del modelo.
    logging.info("Starting evaluation")

    # Carga en la variable `model` la información guardada en el archivo del directorio indicado.
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Se calcula el número de *batches* a partir del tamaño del conjunto de datos de testeo y
    # del tamaño de los *batches*
    num_steps = (params.test_size + 1) // params.batch_size

    # Evalúa el modelo y obtiene las métricas de testeo
    test_metrics = evaluate(model, loss_fn, test_data_iterator, metrics, params, num_steps)

    # Define la ruta del archivo donde se guardarán las métricas de testeo.
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))

    # Guarda en la ruta indicada anteriormente un archivo `json` con las métricas obtenidas.
    utils.save_dict_to_json(test_metrics, save_path)
