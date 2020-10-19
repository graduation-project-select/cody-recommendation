# USAGE
# python recommend.py --dataset datasetTest --input inputClothes/clothes_1.png --closet inputCloset

# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils import paths
from PIL import Image
import tensorflow as tf
import numpy as np
import argparse
import random
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

    # build the label and draw the label on the image
    # labelc = "{}: {:.2f}% ({})".format(labelc, probac[idxc] * 100, correct)
    labelc = "{}: {:.2f}% ({})".format(labelc, probac[idxc] * 100, "")
    labelt = "{}: {:.2f}% ({})".format(labelt, probat[idxt] * 100, "")

    # show the originalImage image
    # print("[INFO] {}".format(labelc))
    # print("[INFO] {}".format(labelt))

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
        # print(str(count) + ": " + "R (" + str(color[0]) + "), G (" + str(color[1]) + "), B (" + str(color[2]) + ")")
        h, s, v = rgbtohsv(color)
        # print("color(hsv) : " + str(h) + ", " + str(s) + ", " + str(v))
        count += 1

        return h, s, v


def classifyAll(img_path):
    color = [0, 0, 0]
    image, image_color, image_original = imagePreprocessing(img_path)
    category, sub_category, texture = classifyAttribute(image)
    print(sub_category)
    color[0], color[1], color[2] = classifyColor(image_color)
    clothes = Clothes(image_original, category, sub_category, texture, color)

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
    def __init__(self, image, category, sub_category, texture, color):
        self.image = image
        self.category = category
        self.sub_category = sub_category
        self.texture = texture
        self.color = color


class Codi:
    def __init__(self):
        self.clothes_list = []

    def addClothes(self, clothes):
        self.clothes_list.append(clothes)


setGPU()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to codi images (i.e., directory of images)")
ap.add_argument("-i", "--input", required=True,
                help="path to select clothes")
ap.add_argument("-c", "--closet", required=True,
                help="path to my closet")
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

input_clothes = classifyAll(args["input"])
print(input_clothes.category + "-" + input_clothes.sub_category + "," + input_clothes.texture)
print("color(hsv) : " + str(input_clothes.color[0]) + ", " + str(input_clothes.color[1]) + ", " + str(
    input_clothes.color[2]))

inputSimilar = []
for codi in codiArr:
    clothes_list = codi.clothes_list
    for clothes in clothes_list:
        if input_clothes.sub_category == clothes.sub_category and input_clothes.texture == clothes.texture:
            if (input_clothes.color[0] >= (clothes.color[0] - 10) or input_clothes.color[0] <= (
                    clothes.color[0] + 10)) and (
                    input_clothes.color[1] >= (clothes.color[1] - 10) or input_clothes.color[1] <= (
                    clothes.color[1] + 10)) and \
                    (input_clothes.color[2] >= (clothes.color[2] - 10) or input_clothes.color[2] <= (
                            clothes.color[2] + 10)):
                # 유사한 상의 찾음
                # clothes.image.show()
                inputSimilar.append(codi)

input_closet = []
closet_clothes_image = os.listdir(args["closet"])
for clothes in closet_clothes_image:
    img_path = args["closet"] + "/" + clothes
    clothes = classifyAll(img_path)
    input_closet.append(clothes)
    print(clothes.category+"-"+clothes.sub_category)

outputSimilar = []
for codi in inputSimilar:
    clothes_list = codi.clothes_list
    recommend_codi = Codi()
    for clothes in clothes_list:
        for closet_clothes in input_closet:
            if closet_clothes.sub_category == clothes.sub_category and closet_clothes.texture == clothes.texture:
                if (closet_clothes.color[0] >= (clothes.color[0] - 10) or closet_clothes.color[0] <= (
                        clothes.color[0] + 10)) and (
                        closet_clothes.color[1] >= (clothes.color[1] - 10) or closet_clothes.color[1] <= (
                        clothes.color[1] + 10)) and \
                        (closet_clothes.color[2] >= (clothes.color[2] - 10) or closet_clothes.color[2] <= (
                                clothes.color[2] + 10)):
                    # 유사한 하의 찾음
                    # closet_clothes.image.show()
                    recommend_codi.addClothes(closet_clothes)
                    break
    outputSimilar.append(recommend_codi)


print("추천받을 수 있는 코디 갯수 : " + str(len(outputSimilar)))

codi_index = 1
for codi in outputSimilar:
    clothes_list = codi.clothes_list
    if not (os.path.isdir("outputClothes/recommend_" + str(codi_index))):
        os.mkdir("outputClothes/recommend_" + str(codi_index))
        input_clothes.image.save("outputClothes/recommend_" + str(codi_index) + "/0.png")
    clothes_index = 1
    for clothes in clothes_list:
        if not input_clothes.category == clothes.category:
            clothes.image.save("outputClothes/recommend_" + str(codi_index) + "/" + str(clothes_index) + ".png")
            clothes_index = clothes_index + 1
    codi_index = codi_index + 1
