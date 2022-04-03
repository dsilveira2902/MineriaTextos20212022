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

En este módulo se define el modelo de la red neuronal, su función de pérdida y su precisión.

 * El modelo de red neuronal define la disposición de las capas. En él se utilizan: una capa de Embeddings,
   una capa LSTM y una capa Linear. Posteriormente, en el método forward se define el procesamiento de la
   información entre las capas.

 * La función de pérdida emplea el cálculo de la entropía cruzada para medir el error.

 * La función de precisión calcula la proporción de predicciones correctas del modelo.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# === Clase `Net` ===
class Net(nn.Module):
    """
        En esta clase se define la estructura de la red neuronal, heredando la estructura proporcionada
        por la clase nn.Module de la librería PyTorch (`torch.nn.Module`).

        Parámetros:

        - `nn.Module`: clase base que proporciona PyTorch para la creación de redes neuronales.
        Modificando esta clase podemos crear redes neuronales propias con diferentes capas y
        propiedades.

    """

    # === Método `__init__` ===
    def __init__(self, params):
        """
            Método constructor de la red neuronal, al cual se llama cada vez que se crea una red
            y define cada una de las capas por las que fluirá la información a través del modelo.

            Parámetros:

            - `params` (`dict`): diccionario en el cual se guardan:

                - `vocab_size`: tamaño del vocabulario del dataset.

                - `embedding_dim`: dimensiones de la matriz de *embeddings*.

                - `lstm_hidden_dim`: dimensiones de la salida de la capa oculta.

                - `number_of_tags`: número de etiquetas de salida posibles.
        """
        # Se realiza una llamada al constructor de la clase padre de la red (clase `nn.Module`)
        super(Net, self).__init__()

        """        
            La primera capa crea una matriz de *embeddings* a partir de las dimensiones de cada vector de embeddings
            (parámetro `params.embedding_dim`) y del tamaño del diccionario de embeddings (parámetro `num_embeddings`).
        """
        self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)

        """
            La segunda capa emplea una red recurrente de tipo LSTM (*Long Short Term Memory*) de manera que utiliza
            los datos actuales y los del instante anterior, proyectando el embedding de tamaño `params.embedding_dim`
            en otro punto del espacio con el tamaño `params.lst_hidden_dim`.  
        """
        self.lstm = nn.LSTM(params.embedding_dim,
                            params.lstm_hidden_dim, batch_first=True)

        # Es una matriz de tamaño lstm_hidden_dim y genera un vector de tamaño number_of_tags

        """
            Esta última capa es de tipo Linear y recibe una matriz que tiene como tamaño `params.lstm_hidden_dim` 
            y genera un vector de tamaño `params.number_of_tags`. 
        """
        self.fc = nn.Linear(params.lstm_hidden_dim, params.number_of_tags)

    # === Método `forward` ===
    def forward(self, s):
        """
            En este método se define cómo se realizará el procesamiento de los tensores de entrada en cada una de las
            capas durante el *paso hacia delante* de cada uno de los *batches*, utilizando los componentes definidos
            en el constructor.

            Parámetros:

            - `s` (`Tensor`): contiene un conjunto de frases con dimensiones nxm, donde n hace referencia al
            tamaño del *batch* y m es la longitud (en tokens) de la frase más larga del batch. En el caso de que
            una frase sea más corta que m, se rellenará con tokens de tipo `PAD` hasta alcanzar esa longitud.

            Return:

            - `out` (`Tensor`): tensor que indica, para cada *token* del *batch* de entrada,
            la probabilidad de pertenecer a una de las posibles clases.
        """

        # Aplica la matriz de embeddings definida en el constructor.
        s = self.embedding(s)

        # Aplica la capa LSTM definida en el constructor.
        s, _ = self.lstm(s)

        # Reorganiza las matrices en memoria, para el correcto funcionamiento del siguiente paso.
        s = s.contiguous()

        # Transforma los datos en una matriz de dos dimensiones (la primera dimensión la calcula automáticamente).
        # s.shape[2] es lo mismo que elegir s.params.lstm_hidden_dim
        # De esta forma, hemos conseguido serializar la matriz

        """
        ### Cambia las dimensiones de la matriz de manera que cada fila contiene un token.
        """
        s = s.view(-1, s.shape[2])

        # Aplica la capa Lineal, de manera que proyecta la nueva proyección de cada *embedding* al
        # número de etiquetas finales.
        s = self.fc(s)

        """
            Aplica la función *SOFTMAX* a la salida de la última capa, de manera que los resultados obtenidos
            para los valores de las etiquetas de la capa final son normalizados a valores entre 0 y 1,
            y así pueden ser interpretados como "probabilidades" de que la palabra pertenezca a cada
            una de las categorías.
        """
        return F.log_softmax(s, dim=1)


# === Método `loss_fn` ===
def loss_fn(outputs, labels):
    """
        Función que calcula la pérdida por entropía cruzada (LCE) a partir
        de los resultados proporcionados por el modelo y las etiquetas de
        los tokens.

        Parámetros:
        - `outputs` (`Tensor`): salida del modelo. Tensor que indica, para cada *token* del *batch* de entrada,
            la probabilidad de pertenecer a una de las posibles clases.

        - `labels` (`Tensor`): contiene los distintos labels de los *tokens* del *batch* de entrada codificados
            mediante el método Label Encoding o -1 si es un token de tipo 'PAD'. Sus dimensiones son nxm,
            donde n hace referencia al tamaño del *batch* y m es la cantidad de *tokens* de la frase
            más larga del *batch*.

        Return:

        - `loss` (`Variable`): pérdida de entropía cruzada (LCE) para todos los tokens del batch excepto aquellos
            que son de tipo 'PAD'.
    """

    # Al pasar como parámetro el valor -1 redimensiona el tensor de labels codificado
    # a un vector de una columna y n filas, siendo el número de filas asignado
    # automáticamente en función del número de elementos.
    labels = labels.view(-1)

    # Genera una máscara de valores de tipo float, en la cual los valores PAD tendrán asignado
    # el valor 0.0 y los demás el valor 1.0.
    mask = (labels >= 0).float()

    # Emplea la aritmética modular para convertir los valores -1 de la etiqueta PAD en valores positivos.
    # A pesar de la modificación de estos valores, la creación de la máscara anterior hace que no afecte
    # al entrenamiento de la red neuronal.
    labels = labels % outputs.shape[1]

    # Calcula el número de tokens totales que no son de tipo PAD
    num_tokens = int(torch.sum(mask))

    # Emplea la máscara para obviar los valores que son de tipo PAD y, mediante el sumatorio,
    # calcula la entropía cruzada de los demás valores de los outputs y los divide entre el
    # número total de tokens.
    return -torch.sum(outputs[range(outputs.shape[0]), labels]*mask)/num_tokens


# === Método `accuracy` ===
def accuracy(outputs, labels):
    """
        Calcula la precisión del modelo a partir de los valores reales y los predichos para todos los tokens
        excepto aquellos que son de tipo PAD.

        Parámetros:
        - `outputs` (`Tensor`): salida del modelo. Tensor que indica, para cada *token* del *batch* de entrada,
            la probabilidad de pertenecer a una de las posibles clases.

        - `labels` (`Tensor`): contiene los distintos labels de los *tokens* del *batch* de entrada codificados
            mediante el método Label Encoding o -1 si es un token de tipo 'PAD'. Sus dimensiones son nxm,
            donde n hace referencia al tamaño del *batch* y m es la cantidad de *tokens* de la frase
            más larga del *batch*.

        Return:
        - `accuracy` (`Variable`): Devuelve el cálculo de la precisión del modelo como un número entre 0 y 1.
    """

    # Redimensiona el tensor de labels codificado a un vector de una columna y n filas,
    # siendo el número de filas asignado automáticamente en función del número de elementos.
    labels = labels.ravel()

    # Genera una máscara de valores de tipo booleano, en la cual los valores PAD tendrán asignado
    # el valor False y los demás el valor True.
    mask = (labels >= 0)

    # Para cada uno de los tokens, devuelve el índice de la clase predicha cuyo valor es más probable.
    outputs = np.argmax(outputs, axis=1)

    # Emplea la máscara para obviar los valores que son de tipo PAD y, mediante la división del número
    # de elementos predichos correctamente entre el número de elementos totales, calcula la precisión
    # del modelo.
    return np.sum(outputs == labels)/float(np.sum(mask))

metrics = {
    'accuracy': accuracy,
}