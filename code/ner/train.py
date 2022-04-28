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


En este módulo se define el entrenamiento de los modelos. Para ello, se crean las funciones `train`
y `train_and_evaluate`. La primera se encarga de entrenar el modelo a partir de: un optimizador,
la función de pérdida, las métricas y otros parámetros. La segunda función utiliza la anterior
para entrenar los modelos y, además, evalúa y guarda: los modelos, los parámetros y las métricas.
"""

# === Importar las librerías ===
import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from tqdm import trange

import utils
import model.net as net
from model.data_loader import DataLoader
from evaluate import evaluate


# === Interfaz de línea de comandos ===

"""
Definimos los argumentos que recibirá nuestro archivo por línea de comandos.
"""
# Creamos el objeto que maneja los argumentos gracias a la librería `argparse`.
parser = argparse.ArgumentParser()
# Parámetro para definir el directorio que contiene el dataset. Por defecto será `data/small`.
parser.add_argument('--data_dir', default='data/small',
                    help="Directory containing the dataset")
# Parámetro para definir el directorio que contiene la configuración del modelo. Por defecto será
# `experiments/base_model`.
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
# Parámetro para definir el archivo que contiene el modelo a cargar. Será `best` o `train`.
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")


# === Función `train` ===
def train(model, optimizer, loss_fn, data_iterator, metrics, params, num_steps):
    """
        Calcula la función de pérdida y, junto con el optimizador, entrena el modelo
        por *batches*. Además, calcula las distintas métricas por cada *batch* para mostrarlas
        de manera informativa en los mensajes de *logging* y ver como avanza el proceso
        de entrenamiento mediante una barra de progreso.

        **Parámetros:**

        - `model`: (`torch.nn.Module`) es un objeto de la clase `Net`, que hace referencia a la red neuronal.

        - `optimizer`: (`torch.optim`) algoritmo de optimización que se utilizará en el entrenamiento.

        - `loss_fn`: ( `function` ) función de pérdida a utilizar para entrenar el modelo en cada uno de los *batches*.

        - `data_iterator`: (`generator`) generador que produce *batches* de datos y etiquetas.

        - `metrics`: (`dict`) diccionario que contiene como clave el nombre de la métrica y como valor la función
          que se usa para calcularla.

        - `params`: (`Params`) objeto de la clase `Params` que contiene las propiedades del conjunto de datos.

        - `num_steps`: (`int`) número de *batches* a ejecutar en el entrenamiento.
    """

    # Ponemos el modelo en modo de entrenamiento.
    model.train()

    # Define una lista que contiene, para algunos de los *batches*, un diccionario
    # con el resultado de la función de pérdida y de cada una de las métricas
    # calculadas.
    summ = []

    # Función que actualiza la media de la función de pérdida forma automática cada vez que se añade un nuevo valor.
    loss_avg = utils.RunningAverage()

    # Función que crea una barra de progreso y muestra el número de *batches* a ejecutar.
    t = trange(num_steps)

    # Por cada *batch*:
    for i in t:

        # Obtenemos los índices de los tokens de las frases y los índices de las etiquetas de las frases.
        train_batch, labels_batch = next(data_iterator)

        # Pasamos la información de los *tokens* al modelo, que ejecutará el `forward` y devolverá las
        # predicciones de etiqueta para cada *token*.
        output_batch = model(train_batch)

        # Se calcula el error entre la predicción del modelo y la etiqueta real de cada *token* mediante la función
        # de pérdida. Normalmente se utiliza la entropía cruzada.
        loss = loss_fn(output_batch, labels_batch)

        # Se resetean los valores del gradiente del modelo, antes de realizar el *backpropagation*, para que no se
        # vean afectados por la ejecución del *batch* anterior.
        optimizer.zero_grad()

        # Emplea la propagación hacia detrás para calcular el gradiente de cada uno de los parámetros.
        loss.backward()

        # Actualiza los parámetros del modelo en función del gradiente calculado.
        optimizer.step()

        # Cuando se cumple esta condición se evalúa el modelo.
        if i % params.save_summary_steps == 0:

            # Procesa los datos de los tensores en la CPU y, posteriormente, los convierte en matrices
            # mediante la función `.numpy()`.
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # Crea un diccionario con los valores de cada una de las métricas para ese *batch*,
            # en el cual, las claves son el nombre de la métrica, y el valor es el resultado
            # de aplicar la función de la métrica.
            summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                             for metric in metrics}

            # Añade el valor de pérdida al diccionario.
            summary_batch['loss'] = loss.item()

            # Añade el diccionario de métricas de un *batch* determinado a la lista.
            summ.append(summary_batch)

        # Actualiza la media de la función de pérdida con el último valor calculado.
        loss_avg.update(loss.item())

        # Actualiza el valor medio de pérdida en la barra de progreso para cada *batch*.
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))

    # Crea un diccionario con el valor medio de cada una de las métricas de todos los *batches*.
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}

    # A partir del diccionario `metrics_mean` genera un string con el valor medio de cada una
    # de las métricas de todos los *batches* separadas por punto y coma.
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())

    # Muestra un mensaje de información en el *logger* con el resultado de las métricas.
    logging.info("- Train metrics: " + metrics_string)


# === Función `train_and_evaluate` ===
def train_and_evaluate(model, train_data, val_data, optimizer, loss_fn, metrics, params, model_dir, restore_file=None):
    """
        Utilizando un modelo precargado o creando uno nuevo, se entrena y evalúa este por épocas con un conjunto de
        entrenamiento y otro de validación. En cada una de estas épocas, se guardará el modelo obtenido y sus métricas
        calculadas y, en caso de que sea el mejor, este también se guardará en un directorio aparte.

        **Parámetros:**

        - `model`: ( `torch.nn.Module` ) es un objeto de la clase `Net`, que hace referencia a la red neuronal.

        - `train_data`: ( `dict` ) diccionario que contiene los datos y las etiquetas de entrenamiento.

        - `val_data`: ( `dict` ) diccionario que contiene los datos y las etiquetas de validación.

        - `optimizer`: (`torch.optim`) algoritmo de optimización que se utilizará en el entrenamiento.

        - `loss_fn`: ( `function` ) función de pérdida a utilizar para entrenar el modelo en cada uno de los *batches*.

        - `metrics`: (`dict`) diccionario que contiene como clave el nombre de la métrica y como valor la función
          que se usa para calcularla.

        - `params`: (`Params`) objeto de la clase `Params` que contiene las propiedades del conjunto de datos.

        - `model_dir`: (`str`) ruta del directorio que contiene la configuración del modelo.

        - `restore_file`: (`str`) ruta del archivo que contiene el modelo a cargar.
    """

    # Si existe el archivo de restauración:
    if restore_file is not None:

        # Se obtiene la ruta completa del archivo de restauración.
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')

        # Muestra un mensaje de información con la ruta del archivo de restauración.
        logging.info("Restoring parameters from {}".format(restore_path))

        # Carga en la variable `model` la información del modelo y en la variable `optimizer` la información
        # del optimizador.
        utils.load_checkpoint(restore_path, model, optimizer)

    # Variable en la que se guardará el mejor valor obtenido para la métrica `accuracy` hasta el momento.
    best_val_acc = 0.0

    # Para cada una de las épocas:
    for epoch in range(params.num_epochs):

        # Muestra un mensaje de información con la época actual del número total de épocas.
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # Se calcula el número de *batches* a partir del tamaño del conjunto de datos de entrenamiento y
        # del tamaño de los *batches*.
        num_steps = (params.train_size + 1) // params.batch_size

        # Crea un generador que lonchea los datos de entrenamiento.
        train_data_iterator = data_loader.data_iterator(
            train_data, params, shuffle=True)

        # Se entrena el modelo con los datos de entrenamiento.
        train(model, optimizer, loss_fn, train_data_iterator,
              metrics, params, num_steps)

        # Se calcula el número de *batches* a partir del tamaño del conjunto de datos de validación y
        # del tamaño de los *batches*.
        num_steps = (params.val_size + 1) // params.batch_size

        # Crea un generador que lonchea los datos de validación.
        val_data_iterator = data_loader.data_iterator(
            val_data, params, shuffle=False)

        # Obtiene las métricas del modelo a partir de los datos de validación.
        val_metrics = evaluate(
            model, loss_fn, val_data_iterator, metrics, params, num_steps)

        # Extrae el valor del `accuracy` para el conjunto de validación de un diccionario de métricas.
        val_acc = val_metrics['accuracy']

        # Comprueba si la nueva métrica de precisión calculada es mejor que la mejor hasta el momento.
        is_best = val_acc >= best_val_acc

        # Se guardan los parámetros del modelo y, si es el mejor modelo hasta el momento, también se
        # guardan en un directorio adicional.
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        # Si el último modelo obtenido es el mejor:
        if is_best:
            # Muestra un mensaje de información indicando que ha encontrado un modelo con una mejor precisión que la
            # del modelo anterior.
            logging.info("- Found new best accuracy")

            # Se actualiza el mejor valor obtenido para la métrica `accuracy`.
            best_val_acc = val_acc

            # Se crea la ruta del archivo JSON con las mejores métricas del conjunto de validación.
            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.json")
            # Se guardan las métricas en la ruta del archivo JSON indicado.
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Se crea la ruta del archivo JSON con las últimas métricas del conjunto de validación.
        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.json")
        # Se guardan las métricas en la ruta del archivo JSON indicado.
        utils.save_dict_to_json(val_metrics, last_json_path)


# === Ejecución principal del archivo ===
if __name__ == '__main__':

    # Argumentos de entrada.
    args = parser.parse_args()

    # Ruta del archivo que contiene la configuración del modelo.
    json_path = os.path.join(args.model_dir, 'params.json')

    # Se comprueba que el archivo se encuentra en la ruta indicada.
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)

    # Se cargan los valores de la configuración del modelo como un objeto de la clase `Params`.
    params = utils.Params(json_path)

    # Indica si el uso de la GPU está disponible.
    params.cuda = torch.cuda.is_available()

    # Establece un valor semilla de manera pseudo-aleatoria para poder reproducir resultados experimentales
    # con los modelos.
    torch.manual_seed(230)

    if params.cuda:
        torch.cuda.manual_seed(230)

    # Crea el archivo de *logs* en la ruta indicada como parámetro.
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Muestra un mensaje de información en el archivo de *logs* indicando la carga de los datos.
    logging.info("Loading the datasets...")

    # Crea un objeto de la clase `DataLoader` en el cual se guarda la información relativa al dataset.
    data_loader = DataLoader(args.data_dir, params)

    # Carga los datos de los directorios `train` y `val`.
    data = data_loader.load_data(['train', 'val'], args.data_dir)

    # Guarda los datos de entrenamiento seleccionados en una variable.
    train_data = data['train']

    # Guarda los datos de validación seleccionados en una variable.
    val_data = data['val']

    # Guarda el tamaño de los datos de entrenamiento seleccionados como una propiedad del conjunto de datos.
    params.train_size = train_data['size']

    # Guarda el tamaño de los datos de validación seleccionados como una propiedad del conjunto de datos.
    params.val_size = val_data['size']

    # Muestra un mensaje de información en el archivo de *logs* indicando el fin de la carga de los datos.
    logging.info("- done.")

    # Comprueba si la GPU está disponible. En caso afirmativo, se crea una copia del modelo en la GPU
    # y, en caso de que no se encuentre disponible, crea el modelo en la CPU.
    model = net.Net(params).cuda() if params.cuda else net.Net(params)

    # Se crea un optimizador de tipo Adam a partir de los parámetros del modelo y de un *learning rate* inicial.
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # Obtiene la función de pérdida del modelo.
    loss_fn = net.loss_fn

    # Obtiene un diccionario con las métricas del modelo.
    metrics = net.metrics

    # Muestra un mensaje de información en el archivo de *logs* indicando el inicio del entrenamiento y el número
    # total de épocas.
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))

    # Entrena y evalúa un modelo por épocas a partir de: los datos de entrenamiento, los datos de validación,
    # el optimizador, la función de pérdida, las métricas y otros parámetros.
    train_and_evaluate(model, train_data, val_data, optimizer, loss_fn, metrics, params, args.model_dir,
                       args.restore_file)
