from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import io

#lecture du fichier
text = open('exemple.txt','r').read().lower()

print('text length:', len(text))

print('-----------------------------------------------')
print(text[:300])
print('-----------------------------------------------')
#trier la liste des char de notre text
chars = sorted(list(set(text)))
print('nbr total de chars:', len(chars))

#utiliser la fnct enumerate pour obtenir les nombres qui representent les
#caracteres
#char-->indice	
char_indices = dict((c, i) for i, c in enumerate(chars))
#cle-->val
indices_char = dict((i, c) for i, c in enumerate(chars))

# definir les donnees d entre x_data et les donnees de sortie y_data et  #convertir char to entier
seq_len = 40
step = 3
x_data = []
y_data = []

for i in range(0, len(text) - seq_len, step):
    #lentre est le caractere actuel+longeur de la sequence
    x_data.append(text[i: i + seq_len])
    #la sortie est le char initial+longeur de la seq
    y_data.append(text[i + seq_len])

print('nb sequences:', len(x_data))

print('Vectorization...')
#converir nos sequences dentree en tableau pr que notre reseau (fct segmoide) #puisse les utiliser pour produire des prob de 0 et 1

x = np.zeros((len(x_data), seq_len, len(chars)), dtype=np.bool)
#(len(x_data), seq_len, len(chars)--> forme de notre tableau
# dtype --> soit 0 ou 1 
y = np.zeros((len(x_data), len(chars)), dtype=np.bool)
for i, sentence in enumerate(x_data):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1

    # y est nos etiquettes pr l'apprentissage
    y[i, char_indices[y_data[i]]] = 1
print('fin')



print('Build model')

model = Sequential()
model.add(LSTM(256, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True)

print('compile model')
model.compile(loss='categorical_crossentropy', optimizer='adam')




def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - seq_len - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + seq_len]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, seq_len, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)




print ('fit model')
print('-----------------------------------------------')
print('generatio befor weight saving')
print('-----------------------------------------------')


model.fit(x, y, epochs=5, batch_size=256, callbacks=[print_callback])


print('-----------------------------------------------')
print('generatio after weight saving')
print('-----------------------------------------------')

#enregistrer les poids apres chque epoch dans un fichier pour les telecharger 
#pour la prochaine itiration 

filepath = "weights_saved.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')


callbacks=[print_callback,checkpoint]

model.fit(x, y, epochs=5, batch_size=256, callbacks=callbacks)


