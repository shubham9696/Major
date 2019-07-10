from keras.models import Model
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import pandas as pd

list_of_labels = ['dresses-and-jumpsuits', 'saree', 'women-jackets-coats', 'women-jeans-jeggings',
                  'women-kurtas-kurtis-suits', 'women-shirts-tops-tees', 'women-shorts-skirts',
                  'women-sweaters-sweatshirts']

class_map = {0: 'dresses-and-jumpsuits', 1: 'saree', 2: 'women-jackets-coats', 3: 'women-jeans-jeggings', 4:
    'women-kurtas-kurtis-suits', 5: 'women-shirts-tops-tees', 6: 'women-shorts-skirts', 7: 'women-sweaters-sweatshirts'}

base_path = "/home/vishakhabanka/Desktop/ImageBasedShopping/data"

model = load_model('best_model.h5')
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(index=175).output)

def main():
    cols = ['id']
    for i in range(1, 2049):
        cols.append(i)

    for label in list_of_labels:
        print(label)
        df = merge_dataframes(base_path + '/train/' + label, cols)
        df.to_csv(base_path + '/train/' + label + "/features22.csv")
        df = merge_dataframes(base_path + '/test/' + label, cols, model)
        df.to_csv(base_path + '/test/' + label + "/features22.csv")


def merge_dataframes(path, cols):
    rows_list = add_rows(path, model, intermediate_layer_model)
    df1 = pd.DataFrame(rows_list, columns=cols)
    df2 = pd.read_csv(path + '/features.csv')
    df = pd.merge(df1, df2, on='id')
    return df


def add_rows(path):
    rows_list = []
    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            label, feature_vector = get_label_and_features(path + '/' + filename)
            dictionary = dict(enumerate(feature_vector, start=1))
            dictionary['id'] = os.path.splitext(filename)[0]
            rows_list.append(dictionary)
    return rows_list


def get_label_and_features(x):
    preds = model._make_predict_function(x)
    y_class = preds.argmax(axis=-1)
    feature_vector = intermediate_layer_model.predict(x)
    feature_vector = feature_vector.flatten()
    return class_map[int(y_class)], feature_vector

def image_to_array(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x


if __name__ == '__main__':
    main()
