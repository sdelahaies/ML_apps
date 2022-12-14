# planets.py

# test_planet.py

import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import os
import random
#from tqdm import tqdm
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout


from PIL import Image, ImageDraw

planets = {
    0: {'name': 'Sun', 'file': 'sun.png', 'code': 'SU'},
    1: {'name': 'Earth', 'file': 'earth.png', 'code': 'EA'},
    2: {'name': 'Mars', 'file': 'mars.png', 'code': 'MA'},
    3: {'name': 'Venus', 'file': 'venus.png', 'code': 'VE'},
    4: {'name': 'Jupiter', 'file': 'jupiter.png', 'code': 'JU'},
    5: {'name': 'Mercury', 'file': 'mercury.png', 'code': 'ME'},
    6: {'name': 'Saturn', 'file': 'saturn.png', 'code': 'SA'},
    7: {'name': 'Neptune', 'file': 'neptune.png', 'code': 'NE'},
    8: {'name': 'Uranus', 'file': 'uranus.png', 'code': 'UR'}
}

bg = []
bg.append(Image.open(r"background/bg1.jpg"))
bg.append(Image.open(r"background/bg2.jpg"))
bg.append(Image.open(r"background/bg3.jpg"))

im_dir = 'data/planets_raw'

for class_id, values in planets.items():
    png_file = Image.open(os.path.join(im_dir, values['file'])).convert('RGBA')
    png_file = png_file.crop(
        (0, 0, np.max(png_file.size), np.max(png_file.size)))
    planets[class_id]['image'] = png_file


def get_bg(bg):
    id_bg = np.random.randint(0, len(bg))
    bg_tmp = bg[id_bg]
    (w, h) = bg_tmp.size
    x1 = np.random.randint(0, w-bg_size)
    y1 = np.random.randint(0, h-bg_size)
    bg_tmp = bg_tmp.crop((x1, y1, x1+bg_size, y1+bg_size))
    return bg_tmp


def put_planet(planet, bg_tmp):
    h = planet.size[0]
    if h >= bg_size:
        scale = 0.4*np.random.random()+0.1
    else:
        scale = 0.7*np.random.random()+0.1

    h = np.int32(scale*h)
    p = planet.resize((h, h))
    h_bg = bg_tmp.size[0]
    x = np.random.randint(0, h_bg-h)
    y = np.random.randint(0, h_bg-h)
    bg_tmp.paste(p, (x, y), mask=p)
    return bg_tmp, x, y, h


bg_size = 800
im_size = 144


def create_example():
    class_id = np.random.randint(0, 9)
    bg_tmp = get_bg(bg)
    plan_im = planets[class_id]['image']
    plan_im = plan_im.rotate(360*np.random.rand())
    img, x, y, h = put_planet(plan_im, bg_tmp)
    img = img.resize((im_size, im_size))
    x1 = np.float32(x)/bg_size
    y1 = np.float32(y)/bg_size
    h = np.float32(h)/bg_size
    return img, class_id, x1, y1, h


def create_model(im_size):
    input_ = Input(shape=(im_size, im_size, 3), name='image')
    x = input_
    for i in range(0, 5):
        n_filters = 2**(4+i)
        x = Conv2D(n_filters, 3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(2)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    class_out = Dense(9, activation='softmax', name='class_out')(x)
    box_out = Dense(3, name='box_out')(x)
    model = tf.keras.models.Model(input_, [class_out, box_out])
    return model


def plot_bounding_box(image, gt_coords, pred_coords, norm=False):
    if norm:
        image *= 255.
        image = image.astype('uint8')
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    [x1, y1, h] = gt_coords
    x1 *= im_size
    y1 *= im_size
    h *= im_size
    draw.rectangle([x1, y1, x1+h, y1+h], outline='green', width=3)
    if len(pred_coords) == 3:
        [x1, y1, h] = pred_coords
        x1 *= im_size
        y1 *= im_size
        h *= im_size
        draw.rectangle([x1, y1, x1+h, y1+h], outline='red', width=3)
    return image  # ,x1,y1,x2,y2


def data_generator():
    while True:
        batch_size = 1
        x_batch = np.zeros((batch_size, im_size, im_size, 3))
        y_batch = np.zeros((batch_size, 9))
        bbox_batch = np.zeros((batch_size, 3))

        for i in range(0, batch_size):
            image, class_id, x1, y1, h = create_example()
            #image,class_id, x1,y1,h=create_example()
            x_batch[i] = np.array(image)/255.
            # x_batch[i]=image/255.
            y_batch[i, class_id] = 1.0
            bbox_batch[i] = np.array([x1, y1, h])
        yield {'image': x_batch}, {'class_out': y_batch, 'box_out': bbox_batch}


def init_model(im_size=144):
    model = create_model(im_size)
    model.load_weights('saved_model_TF/Planets_2_2_weights')
    return model


def test_model(model, test_datagen):
    example, label = next(test_datagen)
    x = example['image']
    y = label['class_out']
    box = label['box_out']

    pred_y, pred_box = model.predict(x)

    pred_coords = pred_box[0]
    gt_coords = box[0]
    pred_class = np.argmax(pred_y[0])
    image = x[0]

    # print(np.argmax(y[0]))
    gt = planets[np.argmax(y[0])]['name']
    pred_class_name = planets[pred_class]['name']

    image = plot_bounding_box(image, gt_coords, pred_coords, norm=True)
    color = 'green' if gt == pred_class_name else 'red'

    #plt.imshow(image)
    #plt.xlabel(f'Pred: {pred_class_name}', color=color)
    #plt.ylabel(f'GT: { gt}', color=color)
    #plt.xticks([])
    #plt.yticks([])


def gen_shot(test_datagen):
    return next(test_datagen)


def detect(shot, model):
    x = shot[0]['image']
    y = shot[1]['class_out']
    box = shot[1]['box_out']

    pred_y, pred_box = model.predict(x)

    pred_coords = pred_box[0]
    gt_coords = box[0]
    pred_class = np.argmax(pred_y[0])
    image = x[0]

    gt = planets[np.argmax(y[0])]['name']
    pred_class_name = planets[pred_class]['name']

    image = plot_bounding_box(image, gt_coords, pred_coords, norm=True)
    return image
