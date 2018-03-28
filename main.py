import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.models.doc2vec import LabeledSentence
from sklearn.model_selection import train_test_split
import random
import csv
import jieba
import numpy as np
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
LabeledSentence = gensim.models.doc2vec.LabeledSentence
trainCSV="./data/train.csv"
testCSV="./data/test.csv"



def get_dataset():
    with open(trainCSV,'r') as infile:
        train_data_raw = list(csv.reader(infile))
        train_y=[int(i[2])-101 for i in train_data_raw]

    with open(testCSV,'r') as infile:
        test_data_raw = list(csv.reader(infile))

    train_y=to_categorical(np.asarray(train_y))
    #将数据分割为训练与测试集
    global x_train_raw
    global x_test_raw
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(train_data_raw, train_y, test_size=0.2)
    x_train=[i[0] for i in x_train_raw]
    x_train_price = [i[1] for i in x_train_raw]
    x_test=[i[0] for i in x_test_raw]
    x_test_price=[i[1] for i in x_test_raw]

    def cleanText(corpus):
        corpus = [list(jieba.cut(z)) for z in corpus]
        return corpus

    x_train = cleanText(x_train)
    x_test = cleanText(x_test)
    test_data = cleanText([i[0] for i in test_data_raw])
    test_price=[i[1] for i in test_data_raw]
    def labelizeReviews(description, label_type):
        labelized = []
        for i,v in enumerate(description):
            label = '%s_%s'%(label_type,i)
            labelized.append(LabeledSentence(v, [label]))
        return labelized

    x_train = labelizeReviews(x_train, 'TRAIN')
    x_test = labelizeReviews(x_test, 'TEST')
    test_data = labelizeReviews(test_data, 'UNSEP')

    return x_train,x_test,y_train, y_test,x_train_price,x_test_price,test_data,test_price

def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)

def train(x_train,x_test,test_data,size = 100,epoch_num=10):
    #实例DM和DBOW模型
    model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
    model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)

    #使用所有的数据建立词典
    #print(np.concatenate((x_train, x_test, unsep_description)))
    model_dm.build_vocab(x_train+x_test+test_data)
    model_dbow.build_vocab(x_train+x_test+test_data)

    #进行多次重复训练，每一次都需要对训练数据重新打乱，以提高精度
    all_train_description = x_train+test_data
    for epoch in range(epoch_num):
        random.shuffle(all_train_description)
        model_dm.train(all_train_description,total_examples=model_dm.corpus_count,epochs=2)
        model_dbow.train(all_train_description,total_examples=model_dbow.corpus_count,epochs=2)

    #训练测试数据集

    for epoch in range(epoch_num):
        random.shuffle(x_test)
        model_dm.train(x_test,total_examples=model_dm.corpus_count,epochs=2)
        model_dbow.train(x_test,total_examples=model_dbow.corpus_count,epochs=2)

    return model_dm,model_dbow


def get_vectors(model_dm,model_dbow):

    #获取训练数据集的文档向量
    train_vecs_dm = getVecs(model_dm, x_train, size)
    train_vecs_dbow = getVecs(model_dbow, x_train, size)
    train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))
    #获取测试数据集的文档向量
    test_vecs_dm = getVecs(model_dm, x_test, size)
    test_vecs_dbow = getVecs(model_dbow, x_test, size)
    test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))


    return train_vecs,test_vecs


def cnn(train_vecs,y_train,test_vecs,y_test):
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    labels_index = {'101': 0, '102': 1, '103': 2, '104': 3, '105': 4, '106': 5, '107': 6, '108': 7, '109': 8, '110': 9}
    embedding_matrix = train_vecs
    embedding_layer = Embedding(8000,
                                200,
                                weights=[embedding_matrix],
                                input_length=200,
                                trainable=False)

    print('Training model.')

    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(200,))
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(len(labels_index), activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    model.fit(train_vecs, y_train,
              batch_size=128,
              epochs=10,
              validation_data=(test_vecs, y_test))
    loss_and_metrics = model.evaluate(test_vecs, y_test, batch_size=128)
    print(loss_and_metrics)


if __name__ == "__main__":
    #设置向量维度和训练次数
    size,epoch_num = 100,20
    #获取训练与测试数据及其类别标注
    x_train, x_test, y_train, y_test, x_train_price, x_test_price, test_data, test_price = get_dataset()
    model_dm,model_dbow = train(x_train,x_test,test_data,size,epoch_num)
    #从模型中抽取文档相应的向量
    train_vecs, test_vecs = get_vectors(model_dm, model_dbow)
    cnn(train_vecs,y_train,test_vecs,y_test)