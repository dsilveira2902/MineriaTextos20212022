import random
import numpy as np
import os
import sys

import torch
from torch.autograd import Variable

import utils 

class DataLoader(object):

    def __init__(self, data_dir, params):
 
        json_path = os.path.join(data_dir, 'dataset_params.json')
        assert os.path.isfile(json_path), "No json file found at {}, run build_vocab.py".format(json_path)
        self.dataset_params = utils.Params(json_path)        
        
        vocab_path = os.path.join(data_dir, 'words.txt')
        self.vocab = {}
        with open(vocab_path) as f:
            for i, l in enumerate(f.read().splitlines()):
                self.vocab[l] = i
        
        self.unk_ind = self.vocab[self.dataset_params.unk_word]
        self.pad_ind = self.vocab[self.dataset_params.pad_word]
                
        tags_path = os.path.join(data_dir, 'tags.txt')
        self.tag_map = {}
        with open(tags_path) as f:
            for i, t in enumerate(f.read().splitlines()):
                self.tag_map[t] = i

        params.update(json_path)

    def load_sentences_labels(self, sentences_file, labels_file, d):

        sentences = []
        labels = []

        with open(sentences_file) as f:
            for sentence in f.read().splitlines():
                s = [self.vocab[token] if token in self.vocab 
                     else self.unk_ind
                     for token in sentence.split(' ')]
                sentences.append(s)
        
        with open(labels_file) as f:
            for sentence in f.read().splitlines():
                l = [self.tag_map[label] for label in sentence.split(' ')]
                labels.append(l)        

        assert len(labels) == len(sentences)
        for i in range(len(labels)):
            assert len(labels[i]) == len(sentences[i])

        d['data'] = sentences
        d['labels'] = labels
        d['size'] = len(sentences)

    def load_data(self, types, data_dir):

        data = {}
        
        for split in ['train', 'val', 'test']:
            if split in types:
                sentences_file = os.path.join(data_dir, split, "sentences.txt")
                labels_file = os.path.join(data_dir, split, "labels.txt")
                data[split] = {}
                self.load_sentences_labels(sentences_file, labels_file, data[split])

        return data

    def data_iterator(self, data, params, shuffle=False):
        """
            Returns a generator that yields batches data with labels. Batch size is params.batch_size. Expires after one
            pass over the data.
        """
        
        order = list(range(data['size']))
        if shuffle:
            random.seed(230)
            random.shuffle(order)

        for i in range((data['size']+1)//params.batch_size):
            batch_sentences = [data['data'][idx] for idx in order[i*params.batch_size:(i+1)*params.batch_size]]
            batch_tags = [data['labels'][idx] for idx in order[i*params.batch_size:(i+1)*params.batch_size]]

            # Obtiene la longitud máxima de una frase, que condiciona la longitud máxima de entrada
            batch_max_len = max([len(s) for s in batch_sentences])

            # Rellena la matriz con todas las frases y añade PADs en los huecos
            # Para esto, creamos una matriz de 1s con el tamaño del minibatch
            # y lo multiplicamos por un 0 o un 1 (el profesor no se acuerda, el valor que tuviera)
            batch_data = self.pad_ind*np.ones((len(batch_sentences), batch_max_len))

            # Creamos una matriz de 1s y ponemos -1 donde no hay etiquetas
            batch_labels = -1*np.ones((len(batch_sentences), batch_max_len))

            # Para cada una de las frases, tenemos una lista de listas, y lo que hacemos
            # es rellenar para cada caso
            for j in range(len(batch_sentences)):
                cur_len = len(batch_sentences[j])
                batch_data[j][:cur_len] = batch_sentences[j]
                batch_labels[j][:cur_len] = batch_tags[j]

            # Mediante .LongTensor() convierte las listas de listas anteriores en tensores
            # Al ser una lista de listas, el tensor será de dimensión 2
            batch_data, batch_labels = torch.LongTensor(batch_data), torch.LongTensor(batch_labels)

            if params.cuda:
                # Metemos los tensores en la GPU mediante .cuda()
                batch_data, batch_labels = batch_data.cuda(), batch_labels.cuda()

            batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)

            # Obtiene 100 datos y 100 etiquetas y las devuelve
            # La siguiente vez que se ejecute vuelve a ejecutar el bucle con los siguientes 100 datos y
            # las siguientes 100 etiquetas
            # Esto produce un "loncheo" de los datos
            yield batch_data, batch_labels
