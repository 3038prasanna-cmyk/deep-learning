import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.preprocessing.sequence import pad_sequences

# Sample toy data
input_texts = ['I love NLP', 'He plays football']
target_texts = [['PRON', 'VERB', 'NOUN'], ['PRON', 'VERB', 'NOUN']]

# Build vocabularies with padding token
PAD_WORD = '<PAD>'
PAD_TAG = '<PAD>'
word_vocab = [PAD_WORD] + sorted(set(word for sent in input_texts for word in sent.split()))
tag_vocab = [PAD_TAG] + sorted(set(tag for tags in target_texts for tag in tags))
word2idx = {w: i for i, w in enumerate(word_vocab)}
tag2idx = {t: i for i, t in enumerate(tag_vocab)}

# Convert texts to sequences
encoder_input_seqs = [[word2idx[word] for word in sent.split()] for sent in input_texts]
decoder_target_seqs = [[tag2idx[tag] for tag in tags] for tags in target_texts]

# Pad sequences
max_encoder_len = max(len(seq) for seq in encoder_input_seqs)
max_decoder_len = max(len(seq) for seq in decoder_target_seqs)
encoder_input_data = pad_sequences(encoder_input_seqs, maxlen=max_encoder_len, padding='post', value=word2idx[PAD_WORD])
decoder_target_data = pad_sequences(decoder_target_seqs, maxlen=max_decoder_len, padding='post', value=tag2idx[PAD_TAG])

# Prepare decoder input data by shifting target sequences to the right, prepend PAD_TAG as start token
decoder_input_data = np.zeros_like(decoder_target_data)
decoder_input_data[:, 1:] = decoder_target_data[:, :-1]
decoder_input_data[:, 0] = tag2idx[PAD_TAG]

# Expand dims for sparse_categorical_crossentropy
decoder_target_data = np.expand_dims(decoder_target_data, -1)

# Define vocabulary sizes
num_encoder_tokens = len(word_vocab)
num_decoder_tokens = len(tag_vocab)
# Model parameters
embedding_dim = 50
latent_dim = 256
# Encoder
encoder_inputs = Input(shape=(None,), name='encoder_inputs')
encoder_embedding = Embedding(num_encoder_tokens, embedding_dim, mask_zero=True, name='encoder_embedding')(encoder_inputs)
encoder_lstm, state_h, state_c = LSTM(latent_dim, return_state=True, name='encoder_lstm')(encoder_embedding)
encoder_states = [state_h, state_c]
# Decoder
decoder_inputs = Input(shape=(None,), name='decoder_inputs')
decoder_embedding = Embedding(num_decoder_tokens, embedding_dim, mask_zero=True, name='decoder_embedding')(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)
# Define and compile model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
# Train
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=2,
    epochs=50,
    verbose=2
)
