from flask import Flask, render_template
from prediction_helpers import *

app = Flask(__name__)

@app.route('/')
def index():

    im1path = get_random_image_path('class1')
    im2path = get_random_image_path('class2')
    im3path = get_random_image_path('class3')
    im4path = get_random_image_path('class4')
    im5path = get_random_image_path('class5')

    impaths = [im1path, im2path, im3path, im4path, im5path]

    training_paths = [str.replace(str_path, 'scaled_images', 'test') for str_path in impaths]

    # get Naive Bayes prediction
    nb_predictions = []
    for im in training_paths:
        nb_predictions.append(get_naive_bayes_prediction(im))

    # get CNN prediction
    CNN_predictions = []
    for im in training_paths:
        CNN_predictions.append(get_CNN_prediction(im))

    # colors for formatting output
    colors = {
        'green': '28fc03',
        'red': 'f47174'}

    correct_predictions = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']

    nb_colors = [colors['green'] if nb_predictions[i] == correct_predictions[i] else colors['red'] for i in range(5)]
    cnn_colors = [colors['green'] if CNN_predictions[i] == correct_predictions[i] else colors['red'] for i in range(5)]

    return render_template('index.html', im1path=im1path, im2path=im2path,
                           im3path=im3path, im4path=im4path, im5path=im5path,
                           nb_predictions=nb_predictions, CNN_predictions=CNN_predictions,
                           nb_colors=nb_colors, cnn_colors=cnn_colors)


