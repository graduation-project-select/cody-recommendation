# USAGE
# python codiToJson.py --dataset datasetTest

# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf
import numpy as np
import argparse
import json
import secrets
import pickle
import cv2
import os

# import for extract color
from sklearn.cluster import KMeans
import utils
import collections


def setGPU():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def imagePreprocessing(imagePath):
    # load the image
    image = cv2.imread(imagePath)
    colorImage = image.copy()
    originalImage = image.copy()

    # pre-process the image for classification
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image, colorImage, originalImage


def classifyAttribute(classify_image):
    # classify the input image
    # print("[INFO] classifying image...")
    # print("[INFO] classifying image - category...")
    probac = modelc.predict(classify_image)[0]
    idxc = np.argmax(probac)
    labelc = lbc.classes_[idxc]
    # print("[INFO] classifying image - texture...")
    probat = modelt.predict(classify_image)[0]
    idxt = np.argmax(probat)
    labelt = lbt.classes_[idxt]
    sub_category = labelc
    texture = labelt

    if sub_category in topList:
        category = "top"
    elif sub_category in bottomList:
        category = "bottom"
    elif sub_category in outerList:
        category = "outer"
    else:
        category = sub_category

    return category, sub_category, texture


def rgbtohsv(color):
    R = color[0] / 255
    G = color[1] / 255
    B = color[2] / 255

    MAX = max(R, G, B)
    MIN = min(R, G, B)

    V = MAX
    if V == 0:
        S = 0
    else:
        S = (V - MIN) / V

    if G == B:
        H = 0
    else:
        if V == R:
            H = (60 * (G - B)) / (V - MIN)
        elif V == G:
            H = 120 + ((60 * (B - R)) / (V - MIN))
        elif V == B:
            H = 240 + ((60 * (R - G)) / (V - MIN))

    if (H < 0):
        H = H + 360

    return round(H), round(S * 100), round(V * 100)


def classifyColor(image_color):
    image_color = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)
    image_color = image_color.reshape((image_color.shape[0] * image_color.shape[1], 3))

    clt = KMeans(n_clusters=3)
    clt.fit(image_color)

    hist = utils.centroid_histogram(clt)

    d = {}
    for (percent, color) in zip(hist, clt.cluster_centers_):
        p = round(percent, 2)
        colors = [int(color[0]), int(color[1]), int(color[2])]  # R: color[0], G: color[1], B: color[2]
        d[p] = colors

    od = collections.OrderedDict(sorted(d.items(), reverse=True))
    # print(od)
    count = 1
    for percent in od:
        if count > 2: break
        color = od[percent]
        # suppose white or black is background
        if (color[0] < 5 and color[1] < 5 and color[2] < 5) or (color[0] > 250 and color[1] > 250 and color[2] > 250):
            # print("background")
            continue
        count += 1

        return color


def classifyAll(img_path):
    image, image_color, image_original = imagePreprocessing(img_path)
    category, sub_category, texture = classifyAttribute(image)
    # print(sub_category)
    color = classifyColor(image_color)
    clothes = Clothes(category, sub_category, texture, color)

    return clothes


def loadCodiData(root_dir, root_dir_list):
    codiArr = []
    for codiID in root_dir_list:
        sub_dir = root_dir + "/" + codiID
        sub_dir_list = os.listdir(sub_dir)
        codi = Codi()
        for img in sub_dir_list:
            img_path = sub_dir + "/" + img
            # print(img_path)
            clothes = classifyAll(img_path)
            codi.addClothes(clothes)
        codiArr.append(codi)
        print(codiID + " : " + str(len(codi.clothes_list)))
    return codiArr


class Clothes:
    def __init__(self, category, sub_category, texture, color):
        self.id = secrets.token_hex(10)
        self.category = category
        self.sub_category = sub_category
        self.texture = texture
        self.color = color

    def clothesToJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True,
                          indent=4)


class Codi:
    def __init__(self):
        self.id = secrets.token_hex(10)
        self.clothes_list = []

    def addClothes(self, clothes):
        self.clothes_list.append(clothes)

    def codiToJSON(self):
        codiJson = json.dumps(self.clothes_list, default=lambda o: o.__dict__, sort_keys=True, indent=4)
        return "\"" + self.id + "\" : {\n\"items\" : " + codiJson + "\n}"


setGPU()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to codi images (i.e., directory of images)")
args = vars(ap.parse_args())

model_c = "model/model_category/category.model"
labelbin_c = "model/model_category/lb.pickle"
model_t = "model/model_texture/texture.model"
labelbin_t = "model/model_texture/lb.pickle"

# load the trained convolutional neural network and the label
# binarizer
print("[INFO] loading network...")
print("[INFO] loading network - category...")
modelc = load_model(model_c)
lbc = pickle.loads(open(labelbin_c, "rb").read())
print("[INFO] loading network - texture...")
modelt = load_model(model_t)
lbt = pickle.loads(open(labelbin_t, "rb").read())

topList = set(["blouse", "longTshirt", "shortTshirt", "sleeveless", "cardigan&vest"])
bottomList = set(["longPants", "shortPants", "skirt"])
outerList = set(["coat", "jacket", "jumper"])

codiImages = os.listdir(args["dataset"])
codiArr = loadCodiData(args["dataset"], codiImages)
print("codi count : " + str(len(codiArr)))

jsonList = ""
for codi in codiArr:
    codiObject = codi.codiToJSON()
    jsonList = jsonList + codiObject
    if not codi == codiArr[-1]:
        jsonList = jsonList + ","
jsonList = jsonList + "\n"

with open("data.json", "w") as json_file:
    json_file.write("{\n\"codi\" : {\n" + jsonList + "\n}\n}")
