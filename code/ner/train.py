
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


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/small',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def train(model, optimizer, loss_fn, data_iterator, metrics, params, num_steps):

    # Ponemos el modelo en modo de entrenamiento. Esto realiza drop out en el modelo y otras cosillas que no se hacen en el modo de evaluación
    model.train()

    summ = []
    loss_avg = utils.RunningAverage()

    # Realiza una barra de progreso mediante caracteres
    # También permite mostrar algunos parámetros que le pasemos
    t = trange(num_steps)
    for i in t:
        train_batch, labels_batch = next(data_iterator) # El iterador devuelve 2 valores

        # Al modelo de la clase Net le pasamos el minibatch de train.
        # En este caso se ejecuta el forward del modelo
        output_batch = model(train_batch)

        # calculamos la pérdida por entropía cruzada a partir de la salida del batch y las etiquetas esperadas
        loss = loss_fn(output_batch, labels_batch)

        # Mientras que no se haga el .zero_grad(), los valores de los pesos no se resetean,
        # sino que se acumulan las derivadas. Lo normal es actualizar los pesos tras ejecutar cada batch
        optimizer.zero_grad()

        # Ya tenemos el error calculado y ese error depende de lo que hay atrás.
        # Empleamos la propagación hacia detrás (backpropagation) para calcular las derivadas
        loss.backward()

        # El optimizador Adam calcula el nuevo training_rate para actualizar los pesos
        optimizer.step()

        if i % params.save_summary_steps == 0:

            # Accede a los datos del tensor y se los lleva a la CPU, y llamando a .numpy() los
            # convierte en una matriz de Numpy.
            # Estos valores (output_batch y labels_batch) los vamos a emplear para calcular el accuracy
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # `metrics` es un diccionario que contiene el nombre de la métrica y la función que la calcula
            # Para cada clave del diccionario, crea un nuevo diccionario que contiene como clave
            # la cadena "accuracy" y como valor el resultado de llamar a la función "accuracy"
            summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                             for metric in metrics}

            # Añade al diccionario la pérdida (entropía cruzada), que es un tensor de 1x1
            # .item() saca el valor del tensor y lo convierte en un número real de Python
            summary_batch['loss'] = loss.item()

            # Añade a una lista las métricas de evaluación
            summ.append(summary_batch)

        loss_avg.update(loss.item())
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))

    # Calcula la media de las métricas
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_data, val_data, optimizer, loss_fn, metrics, params, model_dir, restore_file=None):

    # Si hay que seguir entrenando a partir de un modelo grabado, pues lo carga
    if restore_file is not None:
        # Un modelo se guarda como un diccionario donde la clave suele ser el nombre de la capa
        # y el valor un array con los valores de los pesos
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    # Cuántas épocas queremos recorrer.
    for epoch in range(params.num_epochs):

        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # Calcula el número de pasos en función del tamaño de entrenamiento y el tamaño del batch
        num_steps = (params.train_size + 1) // params.batch_size

        # El data_loader devuelve un iterador
        train_data_iterator = data_loader.data_iterator(
            train_data, params, shuffle=True)

        train(model, optimizer, loss_fn, train_data_iterator,
              metrics, params, num_steps)

        num_steps = (params.val_size + 1) // params.batch_size

        # Después de entrenar, obtenemos un nuevo iterador sobre los datos de evaluación
        val_data_iterator = data_loader.data_iterator(
            val_data, params, shuffle=False)
        val_metrics = evaluate(
            model, loss_fn, val_data_iterator, metrics, params, num_steps)

        # Extrae el valor del accuracy para el conjunto de evaluación de un diccionario de métricas
        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        # Guarda el último y el mejor modelo
        # Se guarda en un fichero binario
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)


if __name__ == '__main__':
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)

    # Lee un fichero en JSON y lo convierte en un objeto de Python
    params = utils.Params(json_path)

    # Miramos si si se detecta la líbreria de CUDA (si está instalada)
    params.cuda = torch.cuda.is_available()

    torch.manual_seed(230)
    if params.cuda:
        # Si tenemos CUDA instalado le enviamos esa misma semilla de forma manual
        torch.cuda.manual_seed(230)

    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    logging.info("Loading the datasets...")

    data_loader = DataLoader(args.data_dir, params)
    # Lee todos los datos en memoria
    data = data_loader.load_data(['train', 'val'], args.data_dir)
    train_data = data['train']
    val_data = data['val']

    params.train_size = train_data['size']
    params.val_size = val_data['size']

    logging.info("- done.")

    # Llamamos al constructor de la clase Net, le pasamos los parámetros con el tamaño de los embeddings, etc.
    # con .cuda mandamos el modelo a la memoria de la GPU
    model = net.Net(params).cuda() if params.cuda else net.Net(params)

    # Optimizador Adam. Aprende un learning_rate diferente con cada peso
    # lr = ... → establece el learning rate inicial
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    loss_fn = net.loss_fn
    metrics = net.metrics # metrics es un diccionario con el nombre de la métrica y la función que la calcula

    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_data, val_data, optimizer, loss_fn, metrics, params, args.model_dir,
                       args.restore_file)
