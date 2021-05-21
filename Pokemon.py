import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
from pathlib import Path

# Join file path 
parent_file = Path(__file__).parent 
file_path = os.path.join(parent_file, 'pokemon_alopez247.csv')

# Read csv file 
df = pd.read_csv(file_path)
# print(df.head())

# Streamline the data that we are looking at
features = ['isLegendary','Generation', 'Type_1', 'Type_2', 'HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed','Color','Egg_Group_1','Height_m','Weight_kg','Body_Style']
df = df[features]

# Convert categorical data into numerical data
def to_dummy(df, categories):
    for category in categories:

        # Get dummy variables (0 or 1) for each entry in category
        column = pd.get_dummies(df[category])
        
        # Add this new column into the dataframe
        df = pd.concat([df, column], axis=1) 

        # Remove category
        df = df.drop(category, axis=1)

    return df

df['isLegendary'] = df['isLegendary'].astype(int)
df = to_dummy(df, ['Egg_Group_1', 'Body_Style', 'Color','Type_1', 'Type_2'])

# def train_test_splitter(DataFrame, column):
#     df_train = DataFrame.loc[df[column] != 1]
#     df_test = DataFrame.loc[df[column] == 1]

#     df_train = df_train.drop(column, axis=1)
#     df_test = df_test.drop(column, axis=1)

#     return(df_train, df_test)

# X_train, X_test = train_test_splitter(df, 'Generation')

# def label_delineator(df_train, df_test, label):
    
#     train_data = df_train.drop(label, axis=1).values
#     train_labels = df_train[label].values
#     test_data = df_test.drop(label,axis=1).values
#     test_labels = df_test[label].values
#     return(train_data, train_labels, test_data, test_labels)

# train_data, y_train, test_data, y_eval = label_delineator(X_train, X_test, 'isLegendary')

from sklearn.model_selection import train_test_split

labels = df.pop('isLegendary')
X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.3, random_state=42)
# y_train = X_train.pop('isLegendary')
# y_test = X_test.pop('isLegendary')

# Convert dataframe to numpy array
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_eval = X_test.to_numpy()
y_eval = y_test.to_numpy()

# Fit and transform model
def data_normalizer(train_data, test_data):
    train_data = preprocessing.MinMaxScaler().fit_transform(train_data)
    test_data = preprocessing.MinMaxScaler().fit_transform(test_data)
    return(train_data, test_data)

train_data, test_data = data_normalizer(X_train, X_test)

# Building Neural Network
shape = train_data.shape

model = keras.Sequential()
model.add(keras.layers.Dense(shape[0], activation='relu', input_shape=[shape[1],]))
model.add(keras.layers.Dense(2, activation='softmax'))
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, y_train, epochs=200)

# Testing the model
loss_value, accuracy_value = model.evaluate(test_data, y_eval)
print(f'Our test accuracy was {accuracy_value}')

# Predicting Pokemons
def predictor(test_data, test_labels, index):
    Dict = {1 : 'Legendary Pokemon', 0 : 'Non-legendary Pokemon'}
    prediction = model.predict(test_data)
    print(type(prediction))
    if np.argmax(prediction[index]) == test_labels[index]:
        print(f'This was correctly predicted to be a \"{Dict[test_labels[index]]}\"!')
    else:
        print(f'This was incorrectly predicted to be a \"{np.argmax(prediction[index])}\". It was actually a \"{test_labels[index]}\".')
        return(prediction)

# Predicting Pokemon 
predictor(test_data, y_eval, 149)