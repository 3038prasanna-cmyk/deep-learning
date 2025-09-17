from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical

# Assume 'corpus' is a list of cleaned Shakespeare text lines
# Example: corpus = ["to be or not to be", "that is the question", ...]

# Initialize and fit the tokenizer on the corpus
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

input_sequences = []

# Create input sequences for the model
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences to ensure equal length
max_len = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

# Split into input (X) and output (y)
X, y = input_sequences[:, :-1], input_sequences[:, -1]

# One-hot encode the output labels
y = to_categorical(y, num_classes=total_words)

# Build the LSTM model
model = Sequential([
    Embedding(total_words, 100, input_length=max_len-1),
    LSTM(150),
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=50, verbose=1)
