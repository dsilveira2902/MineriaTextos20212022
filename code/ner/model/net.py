import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Aquí definimos una red neuronal
class Net(nn.Module):

    def __init__(self, params):
        # Llama al constructor del padre (el padre es nn.Module)
        super(Net, self).__init__()

        # Esto crea una matriz de embeddings con el tamaño del vocabulario (lo recibe como diccionario
        # a través de params.vocab_size). Las dimensiones de la matriz de embeddings se pasan mediante
        # el parámetro params.embedding_dim
        self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)

        # Red recurrente → Proyecta un embedding de tamaño n y lo proyecta en otro punto del espacio
        # Los pesos de la matriz que proyecta también se aprenden
        self.lstm = nn.LSTM(params.embedding_dim,
                            params.lstm_hidden_dim, batch_first=True)

        # Es una matriz de tamaño lstm_hidden_dim y genera un vector de tamaño number_of_tags
        self.fc = nn.Linear(params.lstm_hidden_dim, params.number_of_tags)

    # Método que se va a invocar para decirle a la red neuronal cómo se va procesando la información en cada una
    # de esas capas
    def forward(self, s):

        s = self.embedding(s)

        s, _ = self.lstm(s)

        s = s.contiguous()

        # Transforma los datos en una matriz de dos dimensiones (la primera matriz la calcula automáticamente).
        # s.shape[2] es lo mismo que elegir s.params.lstm_hidden_dim
        # De esta forma, hemos conseguido serializar la matriz
        s = s.view(-1, s.shape[2])

        s = self.fc(s)
        # Los datos están "desmadrados", entonces empleamos la función softmax
        return F.log_softmax(s, dim=1)

# Función de pérdida que calcula la cross-entropy (LCE)
# Recibe las salidas y las etiquetas de las palabras
def loss_fn(outputs, labels):

    labels = labels.view(-1)

    # Esto nos genera un tensor de booleanos, el cual convertiremos a float para poder realizar cálculos
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
