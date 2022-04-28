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

En este módulo se define el procesamiento de los datos. Para ello, primero se cargan las propiedades, el vocabulario y
las etiquetas del dataset, y se guardan en una instancia de la clase `Params`. Mediante los métodos `load_data` y
`load_sentences_labels` se procesan las frases y etiquetas de los conjuntos de datos y se devuelven los índices de
los *tokens* y las etiquetas de cada conjunto. El método `data_iterator` permite iterar sobre los datos previamente
procesados mediante un generador, convirtiéndolos en tensores con los índices de los *tokens* y sus etiquetas.
"""

# === Importar las librerías ===
import random
import numpy as np
import os
import sys

import torch
from torch.autograd import Variable

import utils


# === Clase `DataLoader` ===
class DataLoader(object):
    """
        En esta clase se definen los atributos y las funciones relacionados con el almacenamiento y procesamiento de los datos.
    """

    # === Método `__init__` ===
    def __init__(self, data_dir, params):
        """
            Método constructor de la clase que carga las propiedades, el vocabulario y las etiquetas del dataset
            a partir distintos archivos.

            **Parámetros:**

            - `data_dir`: (`str`) ruta del directorio que contiene el archivo con las propiedades del dataset.

            - `params`: (`Params`) instancia de la clase `Params`, que contiene las propiedades almacenadas
            de un archivo JSON.
        """

        # Ruta del archivo que contiene las propiedades del dataset.
        json_path = os.path.join(data_dir, 'dataset_params.json')
        # Se comprueba que el archivo se encuentra en la ruta indicada.
        assert os.path.isfile(json_path), "No json file found at {}, run build_vocab.py".format(json_path)
        # Se cargan los valores de la configuración del dataset como una propiedad de la clase `DataLoader`.
        self.dataset_params = utils.Params(json_path)

        # Ruta del archivo que contiene el vocabulario con todas las palabras del dataset.
        vocab_path = os.path.join(data_dir, 'words.txt')
        # Diccionario que contendrá todas las palabras del dataset como clave y su índice como valor.
        self.vocab = {}
        # Se leen las palabras del archivo *words.txt* y se extraen al diccionario.
        with open(vocab_path) as f:
            for i, l in enumerate(f.read().splitlines()):
                self.vocab[l] = i

        # Se crean las propiedades de la clase que contendrán las *tokens* que referencian las palabras desconocidas
        # y las de tipo `PAD`.
        self.unk_ind = self.vocab[self.dataset_params.unk_word]
        self.pad_ind = self.vocab[self.dataset_params.pad_word]

        # Ruta del archivo que contiene las etiquetas sintácticas del dataset.
        tags_path = os.path.join(data_dir, 'tags.txt')
        # Diccionario que contendrá todas las etiquetas del dataset como clave y su índice como valor.
        self.tag_map = {}
        # Se leen las etiquetas del archivo *tags.txt* y se extraen al diccionario.
        with open(tags_path) as f:
            for i, t in enumerate(f.read().splitlines()):
                self.tag_map[t] = i

        # Se lee la información del archivo JSON y se almacena en una instancia de la clase `Params`.
        params.update(json_path)

    # === Método `__load_sentences_labels__` ===
    def load_sentences_labels(self, sentences_file, labels_file, d):
        """
            Método que carga las frases y etiquetas del dataset de sus correspondientes archivos. Extrae los índices
            de los *tokens* y sus etiquetas en los vocabularios `vocab` y `tag_map` y almacena esta información en
            un diccionario.

            **Parámetros:**

            - `sentences_file`: (`str`) ruta del archivo *sentences.txt* que contiene las frases con los *tokens* separados por espacios.

            - `labels_file`: (`str`) ruta del archivo *labels.txt* que contiene las etiquetas.

            - `d`: (`dict`) diccionario que contendrá las propiedades de las listas `sentences` y `labels`.
        """

        # Lista que contendrá los índices de los *tokens* de cada una de las frases del archivo *sentences.txt*.
        # Estos índices se extraen del vocabulario (variable `vocab`).
        sentences = []

        # Lista que contendrá los índices de las etiquetas de cada una de las frases del archivo *labels.txt*.
        # Estos índices se extraen del vocabulario (variable `tag_map`).
        labels = []

        # Se abre el contenido del archivo *sentences.txt*.
        with open(sentences_file) as f:
            # Se lee línea a línea el archivo.
            for sentence in f.read().splitlines():
                # Para cada línea, se extraen los *tokens* y se comprueba si estos existen en la variable que contiene el vocabulario.
                # Si existe, se obtiene el índice de dicho *token* en el vocabulario y se añade a la lista.
                s = [self.vocab[token] if token in self.vocab
                     # Si no existe, se añade el índice del *token* `UNK`.
                     else self.unk_ind
                     for token in sentence.split(' ')]
                # Por último, se añade dicha lista a la lista principal.
                sentences.append(s)

        # Se abre el contenido del archivo *labels.txt*.
        with open(labels_file) as f:
            # Se lee línea a línea el archivo.
            for sentence in f.read().splitlines():
                # Para cada línea, se extraen las etiquetas y se añade su índice en el vocabulario de etiquetas.
                l = [self.tag_map[label] for label in sentence.split(' ')]
                # Por último, se añade dicha lista a la lista principal.
                labels.append(l)

        # Comprueba que la longitud de la lista `labels` y de la lista `sentences` es igual.
        assert len(labels) == len(sentences)
        # Para cada elemento, comprueba que el número de *tokens* de la lista `labels` y de la lista `sentences` es igual.
        for i in range(len(labels)):
            assert len(labels[i]) == len(sentences[i])

        # Crea la propiedad `'data'` en el diccionario con el contenido de la lista `sentences`.
        d['data'] = sentences
        # Crea la propiedad `'labels'` en el diccionario con el contenido de la lista `labels`.
        d['labels'] = labels
        # Crea la propiedad `'size'` con la cantidad de elementos que contiene la lista `sentences`, haciendo referencia
        # cada uno de ellos a una frase.
        d['size'] = len(sentences)

    # === Método `load_data` ===
    def load_data(self, types, data_dir):
        """
            Carga las frases y las etiquetas de los conjuntos de datos indicados del directorio `data_dir`, y devuelve
            un diccionario con los índices de los *tokens* y las etiquetas de cada uno de los conjuntos.

            **Parámetros:**

            - `types`: (`list`) lista que contiene uno o varios de los valores `'train'`, `'val'` y `'test'`, indicando los
            datos a cargar.

            - `data_dir`: (`str`) ruta del directorio que contiene el dataset.

            **Return:** `data`: (`dict`) diccionario que contiene los datos cargados, en función de los valores indicados en `types`.
        """

        # Diccionario que contendrá los datos del dataset a cargar.
        data = {}

        # Para cada uno de los posibles valores que puede contener la lista `types`.
        for split in ['train', 'val', 'test']:
            # Si el conjunto actual pertenece a la lista `types`, obtiene las rutas de los archivos con las frases
            # y las etiquetas de cada uno de los conjuntos y almacena en el diccionario los índices de los *tokens*
            # y de las etiquetas de cada conjunto.
            if split in types:
                # Ruta del archivo que contiene las frases.
                sentences_file = os.path.join(data_dir, split, "sentences.txt")
                # Ruta del archivo que contiene las etiquetas.
                labels_file = os.path.join(data_dir, split, "labels.txt")
                data[split] = {}
                # Almacena en el diccionario los indices.
                self.load_sentences_labels(sentences_file, labels_file, data[split])

        return data

    # === Método `data_iterator` ===
    def data_iterator(self, data, params, shuffle=False):
        """
            Carga los datos pasados por parámetro y los convierte en tensores que contienen los índices de los *tokens*
            y sus etiquetas. En función del parámetro que indica el tamaño de *batch*, lonchea estos tensores y
            devuelve un generador.

            **Parámetros:**

            - `data`: (`dict`) diccionario que contiene: los *tokens*, las etiquetas y el número de frases.

            - `params`: (`Params`) instancia de la clase `Params`, que contiene las propiedades almacenadas
              de un archivo JSON.

            - `shuffle`: (`bool`) variable booleana que indica si se deben mezclar los datos o no.

            **Yields:**

            - `batch_data`: (`Variable/Tensor`) matriz de dimensiones **nxm**, donde **n** representa el tamaño del *batch* y **m**
              la longitud máxima de una frase. El contenido de dicha matriz son índices de los *tokens* de las frases.

            - `batch_labels`: (`Variable/Tensor`) matriz de dimensiones **nxm**, donde **n** representa el tamaño del *batch* y **m**
              la longitud máxima de una frase. El contenido de dicha matriz son índices de las etiquetas de las frases.
        """

        # Crea una lista con valores numéricos desde 0 hasta n-1, donde n representa el número de frases.
        order = list(range(data['size']))

        # Si el parámetro `shuffle` es True, se mezclan los valores de la lista `order` de manera pseudoaleatoria.
        if shuffle:
            random.seed(230)
            random.shuffle(order)

        # Calcula el número de *batches* a partir de la cantidad de frases y el tamaño del *batch* e itera por cada uno de los *batches*.
        for i in range((data['size']+1)//params.batch_size):

            # Obtiene las listas con los índices de los *tokens* de las frases de n en n elementos, siendo n el tamaño del *batch*.
            batch_sentences = [data['data'][idx] for idx in order[i*params.batch_size:(i+1)*params.batch_size]]
            # Obtiene las listas con los índices de los etiquetas de las frases de n en n elementos, siendo n el tamaño del *batch*.
            batch_tags = [data['labels'][idx] for idx in order[i*params.batch_size:(i+1)*params.batch_size]]

            # Obtiene la longitud máxima de una frase, que condiciona la longitud máxima de entrada.
            batch_max_len = max([len(s) for s in batch_sentences])

            # Crea una matriz con tamaño **nxm** donde **n** es la longitud de `batch_sentences` y **m** es la longitud
            # máxima de una frase, y sus valores son el índice del *token* de tipo `PAD`.
            batch_data = self.pad_ind*np.ones((len(batch_sentences), batch_max_len))

            # Crea una matriz con tamaño **nxm** donde **n** es la longitud de `batch_sentences` y **m** es la longitud
            # máxima de una frase, y sus valores son -1.
            batch_labels = -1*np.ones((len(batch_sentences), batch_max_len))

            # Rellena las matrices `batch_data` y `batch_labels` con los índices de los *tokens* y de las etiquetas
            # de las frases, respectivamente.
            for j in range(len(batch_sentences)):
                cur_len = len(batch_sentences[j])
                batch_data[j][:cur_len] = batch_sentences[j]
                batch_labels[j][:cur_len] = batch_tags[j]

            # Convierte las matrices `batch_data` y `batch_labels` en tensores de 2 dimensiones.
            batch_data, batch_labels = torch.LongTensor(batch_data), torch.LongTensor(batch_labels)

            # Comprueba si la GPU está disponible.
            if params.cuda:
                # En caso afirmativo, se crea una copia de los tensores `batch_data` y `batch_labels` en la GPU.
                batch_data, batch_labels = batch_data.cuda(), batch_labels.cuda()

            # En versiones anteriores, se pasaban los tensores a tipo `Variable`, para representarlos como nodos de un
            # grafo computacional. Actualmente, esto ya no es necesario.
            batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)

            # En cada iteración del bucle se procesan y se devuelven los elementos de los tensores por *batches*.
            yield batch_data, batch_labels
