from keras.datasets import imdb 
from keras.models import Sequential 
from keras.layers import Embedding, LSTM, Dense 
from keras.preprocessing.sequence import pad_sequences 
# Load dataset 
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000) 
X_train = pad_sequences(X_train, maxlen=100) 
X_test = pad_sequences(X_test, maxlen=100) 
# Model 
model = Sequential([ 
Embedding(10000, 32, input_length=100), 
LSTM(100), 
Dense(1, activation='sigmoid') 
]) 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2) 
