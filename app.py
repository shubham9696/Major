from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import imagenet_utils
from image_color_detection import get_heatmap, get_colors
from keras.models import load_model
from PIL import Image
import numpy as np
import io
import base64
from flask import request
from flask import jsonify
from flask import Flask

base_path = "/home/vishakhabanka/Desktop/ImageBasedShopping/data"
class_map = {0: 'dresses-and-jumpsuits', 1: 'saree', 2: 'women-jackets-coats', 3: 'women-jeans-jeggings', 4:
    'women-kurtas-kurtis-suits', 5: 'women-shirts-tops-tees', 6: 'women-shorts-skirts', 7: 'women-sweaters-sweatshirts'}
query_features = []



app=Flask(__name__)
model=None


def load_keras_model():

    global model
    model=load_model("best_model.h5")
    global graph
    graph = tf.get_default_graph()
    print("Model Loaded!")


def prepare_image(image,target):

    if image.mode != "RGB":
        image = image.convert("RGB")

    image=image.resize(target)
    image=img_to_array(image)
    image=np.expand_dims(image,axis=0)

    return image


print("Loading Keras Model...")
load_keras_model()

def sort_by_colors(nearest, query_color_1, label, K):
    color_distances = get_nearest_colors(base_path + '/train/' + label, 'train', nearest, query_color_1)
    color_distances.extend(get_nearest_colors(base_path + '/test/' + label, 'test', nearest, query_color_1))
    sorted_list = sorted(color_distances, key=itemgetter(1))
    return sorted_list[:K]


def get_nearest_colors(path, train_or_test, nearest, query_color_1):
    df = pd.read_csv(path + '/features_colors.csv')
    rows = df.itertuples(index=False)
    color_distances = []
    for row in rows:
        for img_path, _ in nearest:
            if (train_or_test not in img_path) or (row.id != img_path[img_path.rfind('/') + 1: len(img_path)]):
                continue
            color_distances.append([img_path, colour_distance(query_color_1, row[2])])
    return color_distances


def colour_distance(e1, e2):
    e2 = e2[1: -1]
    e2 = np.fromstring(e2, dtype=float, count=3, sep=' ')
    color1_rgb = sRGBColor(e1[2], e1[1], e1[0])
    color2_rgb = sRGBColor(float(e2[2]), float(e2[1]), float(e2[0]))
    color1_lab = convert_color(color1_rgb, LabColor)
    color2_lab = convert_color(color2_rgb, LabColor)
    delta_e = delta_e_cie2000(color1_lab, color2_lab)
    return delta_e


def get_nearest_neighbours(label, K):
    nearest_k = []
    get_L2_norms(base_path + '/train/' + label, nearest_k)
    get_L2_norms(base_path + '/test/' + label, nearest_k)
    sorted_list = sorted(nearest_k, key=itemgetter(1))
    return sorted_list[:K]


def get_L2_norms(path, nearest_k):
    df = pd.read_csv(path + '/features22.csv')
    rows = df.itertuples(index=False)
    for row in rows:
        calculate_dis(row, path, nearest_k)


def calculate_dis(row, path, nearest_k):
    global query_features
    dis = 0
    for i in range(1, 2049):
        num = float(query_features[i - 1]) - float(row[i + 1])
        dis = dis + (num * num)
    nearest_k.append([path + '/' + row[1], dis])


def get_images(nearest_K, images):
    print(nearest_K)
    for filename, dist in nearest_K:
        images.append(filename + '.jpg')

def get_label_and_features(x):
    preds = model._make_predict_function(x)
    y_class = preds.argmax(axis=-1)
    feature_vector = intermediate_layer_model.predict(x)
    feature_vector = feature_vector.flatten()
    return class_map[int(y_class)], feature_vector


@app.route("/predict",methods=["POST"])
def predict():

    response = {"success":False}

    message = request.get_json(force=True)
    encoded = message["image"]

    decoded = base64.b64decode(encoded)
    image = io.BytesIO(decoded)
    image.seek(0)
    image=Image.open(image)
    processed_image = prepare_image(image,target=(224,224))
    image = image.resize((224, 224))
    predictions = []

    global query_features
    with graph.as_default():
        label, query_features = get_label_and_features(processed_image)
        print("class: " + label)
        query_heatmap = get_heatmap(processed_image)
        query_color_1, query_color_2 = get_colors(query_heatmap, image)  # colors are all in BGR
        nearest_50 = get_nearest_neighbours(label, 501)
        nearest_5 = sort_by_colors(nearest_50, query_color_1, label, 5)
        get_images(nearest_5, predictions)

        response["predictions"] = predictions

        response["success"] = True

    return jsonify(response)






