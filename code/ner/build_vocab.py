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

Este módulo construye los archivos `words.txt`, `tags.txt` y `dataset_params.json` a partir
del contenido del conjunto de datos del directorio `data/`.

- El archivo `words.txt` contiene las palabras extraídas de forma única del conjunto de datos
- El archivo `tags.txt` contiene las etiquetas sintácticas extraídas de forma única del conjunto de datos
- El archivo `dataset_params.json` contiene los parámetros del conjunto de datos
"""
import argparse
from collections import Counter
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--min_count_word', default=1, help="Minimum count for words in the dataset", type=int)
parser.add_argument('--min_count_tag', default=1, help="Minimum count for tags in the dataset", type=int)
parser.add_argument('--data_dir', default='data/small', help="Directory containing the dataset")

# Palabras especiales
PAD_WORD = '<pad>'  # Si una frase tiene 5 palabras y otra 3, la de 3 se rellenará con un pad delante y otro detrás para alcanzar el máximo de 5.
PAD_TAG = 'O'  # Para identificar que para esa palabra no tenemos una etiqueta
UNK_WORD = 'UNK'  # Permite representar una palabra desconocida, esta palabra se trata como cualquier otra palabra, pero tiene su propio embedding en la matriz de embeddings.


# === Función `save_vocab_to_txt_file` ===
def save_vocab_to_txt_file(vocab, txt_path):
    """
        Esta función recorre una lista de palabras extraídas del texto
        de entrenamiento y las escribe cada una en una nueva línea de
        un archivo txt.

        Parámetros:
            * `vocab`: (`list`) vocabulario **de** palabras extraídas del texto de entrenamiento
            * `txt_path`: (`str`) ruta del archivo en el que se guardan las palabras del vocabulario
    """
    with open(txt_path, "w") as f:
        for token in vocab:
            f.write(token + '\n')


# === Función `save_dict_to_json` ===
def save_dict_to_json(d, json_path):
    """
        Esta función guarda la información contenida en un diccionario en un archivo JSON

        Parámetros:
            * `d`: (`dict`) diccionario con las propiedades del conjunto de datos
            * `json_path`: (`str`) ruta del archivo JSON en el que se guardan los parámetros
    """
    with open(json_path, 'w') as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4) # Lo convierte en formato JSON


# === Función `update_vocab` ===
def update_vocab(txt_path, vocab):
    """
        Esta función actualiza el contenido de la estructura de tipo `Counter`
        con el número de apariciones de cada elemento dentro del archivo y
        devuelve el número total de líneas del archivo txt

    Parámetros:

        * `txt_path`: (`str`) ruta del archivo del que se obtiene el contenido a procesar, con una frase por línea
        * `vocab`: (`Counter`) estructura que registra el número de apariciones de cada elemento

    Return:

        * `i`: (`int`) número de líneas del archivo
    """
    with open(txt_path) as f:
        # La función `enumerate()` genera un diccionario con num_linea-1 como clave
        # y el contenido de la línea como valor
        for i, line in enumerate(f):
            vocab.update(line.strip().split(' '))

    return i + 1


if __name__ == '__main__':

    args = parser.parse_args()

    print("Building word vocabulary...")
    words = Counter()
    size_train_sentences = update_vocab(os.path.join(args.data_dir, 'train/sentences.txt'), words)
    size_dev_sentences = update_vocab(os.path.join(args.data_dir, 'val/sentences.txt'), words)
    size_test_sentences = update_vocab(os.path.join(args.data_dir, 'test/sentences.txt'), words)
    print("- done.")

    print("Building tag vocabulary...")
    tags = Counter()
    size_train_tags = update_vocab(os.path.join(args.data_dir, 'train/labels.txt'), tags)
    size_dev_tags = update_vocab(os.path.join(args.data_dir, 'val/labels.txt'), tags)
    size_test_tags = update_vocab(os.path.join(args.data_dir, 'test/labels.txt'), tags)
    print("- done.")

    assert size_train_sentences == size_train_tags
    assert size_dev_sentences == size_dev_tags
    assert size_test_sentences == size_test_tags

    words = [tok for tok, count in words.items() if count >= args.min_count_word]
    tags = [tok for tok, count in tags.items() if count >= args.min_count_tag]

    if PAD_WORD not in words: words.append(PAD_WORD)
    if PAD_TAG not in tags: tags.append(PAD_TAG)
    
    words.append(UNK_WORD)

    print("Saving vocabularies to file...")
    save_vocab_to_txt_file(words, os.path.join(args.data_dir, 'words.txt'))
    save_vocab_to_txt_file(tags, os.path.join(args.data_dir, 'tags.txt'))
    print("- done.")

    sizes = {
        'train_size': size_train_sentences,
        'dev_size': size_dev_sentences,
        'test_size': size_test_sentences,
        'vocab_size': len(words),
        'number_of_tags': len(tags),
        'pad_word': PAD_WORD,
        'pad_tag': PAD_TAG,
        'unk_word': UNK_WORD
    }
    save_dict_to_json(sizes, os.path.join(args.data_dir, 'dataset_params.json'))

    to_print = "\n".join("- {}: {}".format(k, v) for k, v in sizes.items())
    print("Characteristics of the dataset:\n{}".format(to_print))
