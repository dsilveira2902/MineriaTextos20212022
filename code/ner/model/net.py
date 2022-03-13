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

En este módulo se define la el modelo de la red neuronal, su función de pérdida y sus métricas

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# === Clase `Net` ===
class Net(nn.Module):
    """
        En esta clase se define la estructura de la red neuronal, herendando la estructura de
        red neuronal que proporciona PyTorch (torch.nn.Module).

        Parámetros:
            * `nn.Module`: clase base que proporciona PyTorch para la creación de redes neuronales.
            Modificando esta clase podemos crear redes neuronales propias con diferentes capas y
            propiedades.
    """

    # === Método `__init__` ===
    def __init__(self, params):
        """
            Método constructor de la red neuronal, al cual se llama cada vez que se crea una red neuronal
            y define cada una de las capas por las que fluirá la información a través del modelo.

            Parámetros:
                * `params` (`dict`): diccionario en el cual se guardan:
                    * `vocab_size`: tamaño del vocabulario del dataset.
                    * `embedding_dim`: dimensiones de la matriz de *embeddins*.
                    * `lstm_hidden_dim`: dimensiones de la salida de la capa oculta.
                    * `number_of_tags`: número de etiquetas de salida posibles.

        """

        # Se realiza una llamada al constructor de la clase padre de la red.
        super(Net, self).__init__()

        """
            En la primera capa se crea una matriz de *embeddings* del tamaño que se le indica como parámetros 
            al constructor.
        """
        self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)

        """
            En la segunda capa se utiliza una capa tipo LSTM (*Long Short Term Memory*) de manera que utiliza
            los datos actuales y los del instante anterior. Esta capa recibe como parámetro las dimensiones de
            la matriz de *embeddings* y las dimensiones que debe de tener esta misma capa como salida.
        """
        self.lstm = nn.LSTM(params.embedding_dim,
                            params.lstm_hidden_dim, batch_first=True)

        """
            En la última de las capas se utiliza una capa linear, transformando la información de la capa anterior
            a un tensor que tiene por tamaño el número de etiquetas posibles del dataset.
        """
        self.fc = nn.Linear(params.lstm_hidden_dim, params.number_of_tags)

    # === Método `forward` ===
    def forward(self, s):
        """
            En este método se define como se realizará el procesamiento de la información en cada una de las capas
            durante en *paso hacia delante* de cada uno de los *batchs*, utilizando los componentes definidos en el
            constructor.

            Parámetros:
                * `s`: contiene el conjunto de elementos que se van a procesar en el *batch*.

            Return:
                * `out`: tensor que indica, para cada *token* de cada una de las frases a analizar en el *batch*,
                la probabilidad de que pertenezca a cada una de las posibles clases.

        """

        # Aplica la matriz de embeddings definida en el constructor.
        s = self.embedding(s)

        # Aplica la capa LSTM definida en el constructor.
        s, _ = self.lstm(s)

        # Reorganiza las matrices en memoria, para el correcto funcionamiento del siguiente paso.
        s = s.contiguous()

        # Cambia las dimensiones de la matriz de manera que cada fila contiene un token.
        s = s.view(-1, s.shape[2])

        # Aplica la cala Lineal, de manera que proyecta la nueva proyección de cada embedding al
        # número de etiquetas finales.
        s = self.fc(s)

        """
            Aplica el *SOFTMAX* a la salida de la última capa, de manera que los resultados obtenidos
            para los valores de las etiquetas de la capa final son normalizados a valores entre 0 y 1,
            y así pueden ser interpretados como "probabilidades" de que la palabra pertenezca a cada
            una de las categorías.
        """
        return F.log_softmax(s, dim=1)


# === Método `loss_fn` ===
def loss_fn(outputs, labels):

    labels = labels.view(-1)

    mask = (labels >= 0).float()

    labels = labels % outputs.shape[1]

    num_tokens = int(torch.sum(mask))

    return -torch.sum(outputs[range(outputs.shape[0]), labels]*mask)/num_tokens


def accuracy(outputs, labels):

    labels = labels.ravel()

    mask = (labels >= 0)

    outputs = np.argmax(outputs, axis=1)

    return np.sum(outputs == labels)/float(np.sum(mask))


metrics = {
    'accuracy': accuracy,
}
