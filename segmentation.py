from gensim.models import Word2Vec
import csv
import jieba
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

trainCSV="./data/train.csv"
testCSV="./data/test.csv"
MAX_SEQUENCE_LENGTH = 24
MAX_NUM_WORDS = 100000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2

with open(trainCSV, 'r') as infile:
    train_data_raw = list(csv.reader(infile))
with open(testCSV, 'r') as infile:
    test_data_raw = list(csv.reader(infile))

def get_model():
    global sentences1
    global sentences2

    sentences1=[list(jieba.cut(i[0].lower())) for i in train_data_raw]
    sentences2=[list(jieba.cut(i[0].lower())) for i in test_data_raw]
    model = Word2Vec(sentences=sentences1+sentences2,size=200,min_count=0)
    model.save('./model.model')
    print(model.most_similar(['排风']))
    return model

def prepare(model):
    # list of text samples
    labels_index = {'101':0,'102':1,'103':2,'104':3,'105':4,'106':5,'107':6,'108':7,'109':8,'110':9}  # dictionary mapping label name to numeric id
    labels = [int(i[2])-101 for i in train_data_raw]
    labels = to_categorical(np.asarray(labels))# list of label ids

    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    texts=[" ".join(j) for j in sentences1]+[" ".join(j) for j in sentences2]
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences([" ".join(j) for j in sentences1])
    word_index = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print(data.shape)
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]



    num_words=len(word_index)+1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        print(word)
        embedding_matrix[i] = model[word]
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    print('Training model.')

    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 4, activation='relu')(embedded_sequences)
    x = MaxPooling1D(4)(x)
    x = Conv1D(128, 4, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(len(labels_index), activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    model.fit(x_train, y_train,
              batch_size=128,
              epochs=100,
              validation_data=(x_val, y_val))


if __name__ == '__main__':
    model =get_model()
    prepare(model)
