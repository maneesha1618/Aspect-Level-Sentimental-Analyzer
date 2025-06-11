# scripts/train.py

import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import json

train_df = pd.read_csv("data/train.csv")

texts = train_df['Aspect'].values
labels = train_df['Sentiment'].values
label_map = {'negative': 0, 'positive': 1, 'neutral': 2}
y = [label_map[l] for l in labels]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
maxlen = min(max(len(s) for s in sequences), 26)
padded = pad_sequences(sequences, maxlen=maxlen, padding='post')
vocab_size = len(tokenizer.word_index) + 1

inputs = Input(shape=(maxlen,))
x = Embedding(vocab_size, 100)(inputs)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
x = LayerNormalization()(x)
x1 = GlobalMaxPooling1D()(x)
x2 = GlobalAveragePooling1D()(x)
x = Concatenate()([x1, x2])
x = Dense(128)(x)
x = LeakyReLU(0.1)(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
output = Dense(3, activation='softmax')(x)

model = Model(inputs, output)
model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-4), metrics=['accuracy'])
model.summary()

y_cat = to_categorical(y)
model.fit(padded, y_cat, batch_size=128, epochs=10)

model.save("model/aspect_model.keras")

# Save tokenizer
with open("model/tokenizer.json", "w") as f:
    f.write(tokenizer.to_json())

print("✅ Training complete. Model saved.")
