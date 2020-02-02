import string
import tensorflow as tf
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from itertools import chain
import matplotlib.pyplot as plt


# %%


def isplnoun(*line):
    for entry in line:
        if str(entry)[0].isupper():
            if re.match('N 3pl$', str(entry)):
                return True
    return False


raw = pd.read_csv('leefloag/morph_english.flat', sep='\t', lineterminator='\n',
                  names=['pl', 'blk', 'sg', 'pos1', 'pos2', 'pos3', 'pos4'])
raw = raw.values
clean = raw[[not x.startswith(';') for x in raw[:, 0]], :]
clean = np.delete(clean, 1, axis=1)

mask = [isplnoun(*x) for x in clean]
clean = clean[mask, :]
clean = clean[:, :2]
clean = clean[~pd.isnull(clean).any(axis=1)]
clean = np.array([[re.sub('[^a-z]', '', x.strip().lower()), re.sub('[^a-z]', '', y.strip().lower())] for x, y in clean])
clean = np.array([x for x in clean if len(x[0]) < 15])
# %%

charset = set()
for x in clean.flatten():
    for y in x:
        charset.add(y)
mapping = {c: i for i, c in enumerate(charset)}
mapping['?'] = len(mapping)
invmap = {i: c for c, i in mapping.items()}
lens = np.vectorize(len)(clean.transpose().flatten())
maxlen = max(lens)


def map_encode(*words, code=mapping, l=maxlen):
    output = []
    for word in words:
        xpand = list(word)
        padval = l - len(xpand)
        xpand[len(xpand):] = ['?'] * padval
        mapped = [code[y] for y in xpand]
        output.append(mapped)
    return np.array(output)


def map_decode(*ctext, code=invmap, strip_eofs=True):
    output = []
    for word in ctext:
        dcoded = [code[x] for x in word]
        dcoded = ''.join(dcoded)
        if strip_eofs:
            dcoded = dcoded.strip('?')
            dcoded = dcoded.split('?')[0]
        output.append(dcoded)
    return np.array(output)


# %%

in_train = tf.keras.utils.to_categorical(np.array(map_encode(*clean.transpose()[1])))
out_train = tf.keras.utils.to_categorical(np.array(map_encode(*clean.transpose()[0])))

# %% monolayer

minput = tf.keras.layers.Input(shape=(in_train.shape[1], in_train.shape[2]), name='main_input')
f = tf.keras.layers.Flatten()(minput)
h = tf.keras.layers.Dense(512, activation='relu')(f)

out_layers = []
for letter in range(in_train.shape[1]):
    out_layers.append(tf.keras.layers.Dense(in_train.shape[2], activation='softmax')(h))

slayermodel = tf.keras.models.Model(inputs=[minput], outputs=out_layers)

slayermodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# %%

better_outs = [out_train[:, x, :] for x in range(out_train.shape[1])]
slayermodel.fit(in_train, better_outs, epochs=100)

# %%

tests = ['coen', 'jessica', 'mouse', 'smeej', 'door', 'deer', 'moose', 'jeff',
         'gerpgork', 'leefloag', 'elf', 'child', 'focus', 'torus', 'house', 'frilg', 'walrus']


def make_answer_set(ins, model):
    outs = []
    for word in ins:
        y_test = model.predict(tf.keras.utils.to_categorical(map_encode(word)))
        outs.append(map_decode(np.argmax(y_test, axis=2).flatten())[0])
    print('|Input|Output|')
    for i, o, in zip(ins, outs):
        print(f'|{i}|{o}|')


# %% bilayer

minput = tf.keras.layers.Input(shape=(in_train.shape[1], in_train.shape[2]), name='main_input')
f = tf.keras.layers.Flatten()(minput)
h = tf.keras.layers.Dense(128, activation='relu')(f)
h = tf.keras.layers.Dense(128, activation='relu')(h)

out_layers = []
for letter in range(in_train.shape[1]):
    out_layers.append(tf.keras.layers.Dense(in_train.shape[2], activation='softmax')(h))

dlayermodel = tf.keras.models.Model(inputs=[minput], outputs=out_layers)

dlayermodel.compile(loss='categorical_crossentropy', optimizer='adam')

better_outs = [out_train[:, x, :] for x in range(out_train.shape[1])]
dlayermodel.fit(in_train, better_outs, epochs=100)

# %% trilayer

minput = tf.keras.layers.Input(shape=(in_train.shape[1], in_train.shape[2]), name='main_input')
f = tf.keras.layers.Flatten()(minput)
h = tf.keras.layers.Dense(512, activation='relu')(f)
h = tf.keras.layers.Dense(512, activation='relu')(h)
h = tf.keras.layers.Dense(512, activation='relu')(h)

out_layers = []
for letter in range(in_train.shape[1]):
    out_layers.append(tf.keras.layers.Dense(in_train.shape[2], activation='softmax')(h))

tlayermodel = tf.keras.models.Model(inputs=[minput], outputs=out_layers)

tlayermodel.compile(loss='categorical_crossentropy', optimizer='adam')

better_outs = [out_train[:, x, :] for x in range(out_train.shape[1])]
tlayermodel.fit(in_train, better_outs, epochs=100)

# %% nillayer

minput = tf.keras.layers.Input(shape=(in_train.shape[1], in_train.shape[2]), name='main_input')
f = tf.keras.layers.Flatten()(minput)

out_layers = []
for letter in range(in_train.shape[1]):
    out_layers.append(tf.keras.layers.Dense(in_train.shape[2], activation='softmax')(f))

nohmodel = tf.keras.models.Model(inputs=[minput], outputs=out_layers)

nohmodel.compile(loss='categorical_crossentropy', optimizer='adam')

better_outs = [out_train[:, x, :] for x in range(out_train.shape[1])]
nohmodel.fit(in_train, better_outs, epochs=100)


# %% md

# letter-by-letter model

# %%

def genngrams(word, n):
    wordpad = '?' * (n) + word + '?' * (n-1)
    ngrams = zip(*[wordpad[i:] for i in range(n + 1)])
    for j in ngrams:
        gram = j[:n]
        nextchar = j[-1]
        yield ''.join(gram), nextchar


lblins = []
lblprevs = []
lblouts = []

for o, i in clean:
    for gram3, nextchar in genngrams(o, 5):
        lblins.append(i)
        lblprevs.append(gram3)
        lblouts.append(nextchar)


lblins = tf.keras.utils.to_categorical(np.array(map_encode(*lblins)))
lblprevs = tf.keras.utils.to_categorical(np.array(map_encode(*lblprevs, l=5)))
lblouts = tf.keras.utils.to_categorical(np.array(map_encode(*lblouts, l=1)))

#%%
# 1x1layermodel

minput = tf.keras.layers.Input(shape=(lblins.shape[1], lblins.shape[2]), name='main_input')
mf = tf.keras.layers.Flatten()(minput)

pinput = tf.keras.layers.Input(shape=(lblprevs.shape[1], lblprevs.shape[2]), name='prev_output')
pf = tf.keras.layers.Flatten()(pinput)

f = tf.keras.layers.concatenate([pf, mf])

h = tf.keras.layers.Dense(512, activation='relu')(f)


outlayer = tf.keras.layers.Dense(27, activation='softmax')(h)

lblmonolayer = tf.keras.models.Model(inputs=[minput, pinput], outputs=[outlayer])

lblmonolayer.compile(loss='categorical_crossentropy', optimizer='adam')

#%%

lblmonolayer.fit([lblins, lblprevs], lblouts, epochs=25)

#%%


def lbl_pluralize(sing, model, printchars=False):
    i = tf.keras.utils.to_categorical(map_encode(sing))
    current = '???'
    nchar = ''
    last3 = tf.keras.utils.to_categorical(map_encode(current, l=5))
    while nchar != '?':
        ou = model.predict([i, last3])
        nchar = map_decode([np.argmax(ou)], strip_eofs=False)[0]
        current += nchar
        if printchars:
            print(nchar)
        last3 = tf.keras.utils.to_categorical(map_encode(current[-5:], l=5), num_classes=27)
    return current.strip('?')


def make_lbl_table(words, model):
    print('|Input|Output|')
    for word in words:
        out = lbl_pluralize(word, model)
        print(f'|{word}|{out}|')


#%%


minput = tf.keras.layers.Input(shape=(lblins.shape[1], lblins.shape[2]), name='main_input')
mf = tf.keras.layers.Flatten()(minput)

pinput = tf.keras.layers.Input(shape=(lblprevs.shape[1], lblprevs.shape[2]), name='prev_output')
pf = tf.keras.layers.Flatten()(pinput)

f = tf.keras.layers.concatenate([pf, mf])

h = tf.keras.layers.Dense(512, activation='relu')(f)
h = tf.keras.layers.Dense(512, activation='relu')(h)


outlayer = tf.keras.layers.Dense(27, activation='softmax')(h)

lblbilayer = tf.keras.models.Model(inputs=[minput, pinput], outputs=[outlayer])

lblbilayer.compile(loss='categorical_crossentropy', optimizer='adam')

lblbilayer.fit([lblins, lblprevs], lblouts, epochs=25)

#%%
# trilayer

minput = tf.keras.layers.Input(shape=(lblins.shape[1], lblins.shape[2]), name='main_input')
mf = tf.keras.layers.Flatten()(minput)

pinput = tf.keras.layers.Input(shape=(lblprevs.shape[1], lblprevs.shape[2]), name='prev_output')
pf = tf.keras.layers.Flatten()(pinput)

f = tf.keras.layers.concatenate([pf, mf])

h = tf.keras.layers.Dense(512, activation='relu')(f)
h = tf.keras.layers.Dense(512, activation='relu')(h)
h = tf.keras.layers.Dense(512, activation='relu')(h)


outlayer = tf.keras.layers.Dense(27, activation='softmax')(h)

lbltrilayer = tf.keras.models.Model(inputs=[minput, pinput], outputs=[outlayer])

lbltrilayer.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

lbltrilayer.fit([lblins, lblprevs], lblouts, epochs=3, validation_split=.25)

#%%

# bigbrain

minput = tf.keras.layers.Input(shape=(lblins.shape[1], lblins.shape[2]), name='main_input')
mf = tf.keras.layers.Flatten()(minput)

pinput = tf.keras.layers.Input(shape=(lblprevs.shape[1], lblprevs.shape[2]), name='prev_output')
pf = tf.keras.layers.Flatten()(pinput)

f = tf.keras.layers.concatenate([pf, mf])

h = tf.keras.layers.Dense(512, activation='relu')(f)
for _ in range(9):
    h = tf.keras.layers.Dense(512, activation='relu')(h)


outlayer = tf.keras.layers.Dense(27, activation='softmax')(h)

lbltrilayer = tf.keras.models.Model(inputs=[minput, pinput], outputs=[outlayer])

lbltrilayer.compile(loss='mean_absolute_error', optimizer='adam')

lbltrilayer.fit([lblins, lblprevs], lblouts, epochs=5, validation_split=.2)

