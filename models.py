from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, Dropout
from tensorflow.keras.regularizers import l2
import pandas as pd


class Shallom:
    def __init__(self, *, settings, num_entities, num_relations):
        self.settings = settings
        self.model = Sequential()
        self.model.add(Embedding(num_entities, self.settings['embedding_dim'],
                                 input_length=2, activity_regularizer=l2(self.settings['reg'])))
        self.model.add(Flatten())
        self.model.add(Dropout(self.settings['input_dropout']))
        self.model.add(Dense(self.settings['embedding_dim'] * self.settings['hidden_width_rate'], activation='relu',
                             activity_regularizer=l2(self.settings['reg'])))
        self.model.add(Dropout(self.settings['hidden_dropout']))
        self.model.add(Dense(num_relations, activation='sigmoid'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

    def fit(self, X, y):
        self.model.fit(X, y, batch_size=self.settings['batch_size'], epochs=self.settings['epochs'],
                       use_multiprocessing=True, verbose=1, shuffle=True)

    def predict(self, X):
        return self.model.predict(X)

    def embeddings_save_csv(self, entity_idx, path):
        emb = self.model.layers[0].get_weights()[0]
        df = pd.DataFrame(emb, index=list(entity_idx.keys()))
        df.to_csv(path +'/embeddings.csv')
