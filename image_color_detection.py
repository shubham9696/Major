from sklearn.cluster import KMeans
import numpy as np
import cv2
from keras.models import load_model, Model
import scipy
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
import os
import pandas as pd
from tqdm import tqdm

list_of_labels = ['women-shirts-tops-tees', 'women-kurtas-kurtis-suits', 'women-jeans-jeggings',
                  'women-sweaters-sweatshirts', 'saree']

base_path = "/home/vishakhabanka/Desktop/ImageBasedShopping/data"


def get_ResNet_Model():
    model = load_model('best_model.h5')
    all_amp_layer_weights = model.layers[-1].get_weights()[0]
    ResNet_model = Model(inputs=model.input, outputs=(model.layers[-4].output, model.layers[-1].output))
    return ResNet_model, all_amp_layer_weights


ResNet_model, all_amp_layer_weights = get_ResNet_Model()


def main():
    for label in list_of_labels:
        print(label)
        save_dataframe_for_given_path(base_path + '/train/' + label)
        save_dataframe_for_given_path(base_path + '/test/' + label)


def save_dataframe_for_given_path(path):
    rows_list = []
    for filename in tqdm(os.listdir(path)):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            img_path = path + '/' + filename
            heatmap = get_heatmap(img_path)
            color1, color2 = get_colors(heatmap, img_path)
            arr = [os.path.splitext(filename)[0], color1, color2]
            rows_list.append(arr)
    df = pd.DataFrame(rows_list, columns=['id', 1, 2])
    df.to_csv(path + "/features_colors.csv")


def get_heatmap(img):
    CAM = ResNet_CAM(img)
    min_ = np.min(CAM)
    max_ = np.max(CAM)
    heatmap = (255 * (CAM - min_) / (max_ - min_)).astype(np.uint8)
    return heatmap


def ResNet_CAM(img):
    last_conv_output, pred_vec = ResNet_model.predict(preprocess_input(img))
    last_conv_output = np.squeeze(last_conv_output)
    pred = np.argmax(pred_vec)
    mat_for_mult = scipy.ndimage.zoom(last_conv_output, (32, 32, 1), order=1)  # dim: 224 x 224 x 2048
    amp_layer_weights = all_amp_layer_weights[:, pred]  # dim: (2048,)
    final_output = np.dot(mat_for_mult.reshape((224 * 224, 2048)), amp_layer_weights).reshape(224,
                                                                                              224)  # dim: 224 x 224
    return final_output


def pretrained_path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)


def get_colors(heatmap, img):
    colorToBeIgnored = heatmap[0][0]

    for i in range(0, 224):
        for j in range(0, 224):
            if colorToBeIgnored >= 120 and heatmap[i][j] > 80:  # todo: OR should it be < 180??? white is 255
                img[i][j] = (0, 255, 255)
            elif colorToBeIgnored < 120 and heatmap[i][j] < 150:
                img[i][j] = (0, 255, 255)

    height, width, dim = img.shape
    img_vec = np.reshape(img, [height * width, dim])

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(img_vec)

    unique_l, counts_l = np.unique(kmeans.labels_, return_counts=True)
    sort_ix = np.argsort(counts_l)
    sort_ix = sort_ix[::-1]

    return kmeans.cluster_centers_[sort_ix][1], kmeans.cluster_centers_[sort_ix][2]


if __name__ == '__main__':
    main()
