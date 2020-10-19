import os
from PIL import Image

input_dir = "codi"
output_dir = "codiToFolder"
images = []
images_name = []

def codiToFolder():
    codiImages = os.listdir(input_dir)
    for codi in codiImages:
        images_name.append(codi)
        image = Image.open(input_dir + '/' + codi)
        images.append(image)

    for i in range(0, len(images), 1):
        codiID = images_name[i].split('_')[0]
        codiItemNum = images_name[i].split('_')[1]
        print(codiID)
        if not (os.path.isdir(output_dir + '/' + codiID)):
            os.mkdir(output_dir + '/' + codiID)
        images[i].save(output_dir + '/' + codiID + '/' + codiItemNum)


codiToFolder()