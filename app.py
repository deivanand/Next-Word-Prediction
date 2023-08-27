import re 
import pickle
import time  
from nltk.tokenize import word_tokenize
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from flask import Flask, request, render_template

#Attention Model for classification 
class attention(Layer):
    def init(self):
        super(attention,self).__init__()
    def build(self,input_shape):
        self.W=self.add_weight(name='att_weight',shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name='att_bias',shape=(input_shape[-2],1),initializer="zeros")        
        super(attention, self).build(input_shape)
    def call(self,x):
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        return K.sum(output, axis=1)

def extra_space(text):
    new_text= re.sub("\s+"," ",text)
    return new_text

def sp_charac(text):
    new_text=re.sub("[^0-9A-Za-z ]", "" , text)
    return new_text

def tokenize_text(text):
    new_text=word_tokenize(text)
    return new_text


app = Flask(__name__)

with open('len_tokens_ATT6.pickle', 'rb') as handle:
    length_tokens_6 = pickle.load(handle)
    
with open('len_tokens_ATT4.pickle', 'rb') as handle:
    length_tokens_4 = pickle.load(handle)
 
with open('len_tokens_ATT2.pickle', 'rb') as handle:
    length_tokens_2 = pickle.load(handle)

file="lstm_att_len6.hdf5"
model_len6 = load_model(file ,custom_objects={'attention': attention})
model_len6.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

file="lstm_att_len4.hdf5"
model_len4 = load_model(file, custom_objects={'attention': attention})
model_len4.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

file="lstm_att_len2.hdf5"
model_len2 = load_model(file , custom_objects={'attention': attention})
model_len2.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

@app.route('/')
def my_form():
    return render_template('my-form.html')

@app.route('/', methods=['POST'])
def predict_next():
    text = request.form['text']
    if not text:
        return render_template('my-form.html', error="Please enter a new word / sentence!")
    start= time.time()
    cleaned_text=extra_space(text)
    cleaned_text=sp_charac(cleaned_text)
    tokenized=tokenize_text(cleaned_text)
    line = ' '.join(tokenized)
    pred_words = []
    if len(tokenized)==1:
        encoded_text = length_tokens_2.texts_to_sequences([line])
        pad_encoded = pad_sequences(encoded_text, maxlen=1, truncating='pre')
        for i in (model_len2.predict(pad_encoded)[0]).argsort()[-3:][::-1]:
            pred_word = length_tokens_2.index_word[i]
            pred_words.append(text + " " + pred_word)
    elif len(tokenized) < 4:
        encoded_text = length_tokens_4.texts_to_sequences([line])
        pad_encoded = pad_sequences(encoded_text, maxlen=3, truncating='pre')
        for i in (model_len4.predict(pad_encoded)[0]).argsort()[-3:][::-1]:
            pred_word = length_tokens_4.index_word[i]
            pred_words.append(text + " " + pred_word)
    else:
        encoded_text = length_tokens_6.texts_to_sequences([line])
        pad_encoded = pad_sequences(encoded_text, maxlen=5, truncating='pre')
        for i in (model_len6.predict(pad_encoded)[0]).argsort()[-3:][::-1]:
            pred_word = length_tokens_6.index_word[i]
            pred_words.append(text + " " + pred_word)
    print('Time taken: ',time.time()-start)
    return render_template('my-form.html', pred_words=pred_words)

if __name__ == '__main__':
    app.run()


