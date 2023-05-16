print("[INFO]: Importing packages")
# general tools
import os
import numpy as np
import argparse
import tensorflow as tf
from tqdm import tqdm

# data loading
from utils import (load_style_img, crop_resize, build_model, get_relpaths, plot_imgs, plot_history)
from tensorflow.keras.utils import load_img

# image processsing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

# style transfer model
import tensorflow_hub as hub

#scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.metrics import classification_report

# terminal parsing function
def input_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, default=None, help="file name of style image including extension")
    parser.add_argument("--flip_shift", action="store_true", help="apply classic data augmentation methods")

    args = parser.parse_args()

    return(args)

def main(style, flip_shift):
    
    # Load data
    print("[INFO]: Loading data")
    train_cat_paths = get_relpaths(sub="training_set", subsub="cats")
    train_dog_paths = get_relpaths(sub="training_set", subsub="dogs")
    test_cat_paths = get_relpaths(sub="test_set", subsub="cats")
    test_dog_paths = get_relpaths(sub="test_set", subsub="dogs")

    X_train_cat = np.array([np.array(load_img(fname, target_size=(224, 224))) for fname in train_cat_paths])
    X_train_dog = np.array([np.array(load_img(fname, target_size=(224, 224))) for fname in train_dog_paths])
    X_test_cat = np.array([np.array(load_img(fname, target_size=(224, 224))) for fname in test_cat_paths])
    X_test_dog = np.array([np.array(load_img(fname, target_size=(224, 224))) for fname in test_dog_paths])

    # Concatenate arrays
    X_train = np.concatenate([X_train_cat, X_train_dog])
    X_test = np.concatenate([X_test_cat, X_test_dog])
    del X_train_cat, X_train_dog, X_test_cat, X_test_dog

    # Normalize
    X_train = X_train.astype(np.float32) / 255.
    X_test = X_test.astype(np.float32) / 255.

    # Make y labels, cat = 0, dog = 1
    y_train = [0 for n in range(len(X_train_cat))] + [1 for n in range(len(X_train_dog))]
    y_test = [0 for n in range(len(X_test_cat))] + [1 for n in range(len(X_test_dog))]

    if style != None:

        # Change train data to format compatible with style transfer model
        X_train_t = tf.constant(X_train)

        # Load style image
        style_path = os.path.join("..", "styles", f"{style}")
        style_img = load_style_img(style_path)
        style_img = crop_resize(style_img)

        # Blur style image
        style_img = tf.nn.avg_pool(style_img, ksize=[3,3], strides=[1,1], padding='SAME')

        # Load style transfer model
        print("[INFO]: Loading style transfer model")
        hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
        hub_module = hub.load(hub_handle)

        # Generate stylized images in batches to not overload memory
        print("[INFO]: Generating stylized images")
        batch_size_stylize = 100 

        num_samples = len(X_train_t)
        num_batches = num_samples // batch_size_stylize

        stylized_X_train = []
        for i in tqdm(range(num_batches)):
            start = i * batch_size_stylize
            end = start + batch_size_stylize
            batch = X_train_t[start:end]
            outputs = hub_module(batch, style_img)
            stylized_X_train.extend(outputs[0])

        # Convert stylized images back to original X_train format
        stylized_X_train = np.array(stylized_X_train)

        # Combine
        X_train = np.concatenate([X_train, stylized_X_train])
        y_train = y_train + y_train

        # Save figure of style transfer examples
        style_trim = style.replace(".png","")
        plot_imgs(images=[X_train[0], np.array(style_img[0]), X_train[8000]],
                  fname=f"{style_trim}_cat.png")
        plot_imgs(images=[X_train[4000], np.array(style_img[0]), X_train[12000]],
                  fname=f"{style_trim}_dog.png")
    
    # Shuffle data
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)

    # Define image data generators
    if flip_shift:
        train_generator = ImageDataGenerator(horizontal_flip=True,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            preprocessing_function=preprocess_input, 
                                            validation_split=0.2)
        train_generator.fit(X_train)
    else:
        train_generator = ImageDataGenerator(preprocessing_function=preprocess_input,
                                            validation_split=0.2)
        train_generator.fit(X_train)
    
    test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator.fit(X_test)

    # Build model
    model = build_model()
    
    # Define appropriate batch sizes
    if style != None:
        batch_size_train = 64
        batch_size_val = 16
    else:
        batch_size_train = 32
        batch_size_val = 8 

    # Fit model
    print("[INFO]: Model training")
    H = model.fit(train_generator.flow(X_train, y_train, 
                                       batch_size=batch_size_train, 
                                       subset='training', 
                                       seed=5), 
                  validation_data = train_generator.flow(X_train, y_train, 
                                                        batch_size=batch_size_val, 
                                                        subset = "validation", 
                                                        seed=5),
                  epochs=10,
                  verbose=1)
    
    # Save model and history plot
    if style != None:
        model_name = f"style_{style_trim}"
        plot_name = f"history_style_{style}"
    elif flip_shift:
        model_name = "flip_shift"
        plot_name = "history_flip_shift_aug.png"
    else: 
        model_name = "no_aug"
        plot_name = "history_no_aug.png"
    
    model_path = os.path.join("..","models",f"model_{model_name}.SavedModel")
    model.save(model_path)
    
    plot_history(H, epochs=10, title=plot_name)

    # Predict
    print("[INFO]: Predicting")
    predictions = model.predict(test_generator.flow(X_test,
                                                    batch_size=1,
                                                    shuffle=False))
    predictions = [1 if pred >= 0.5 else 0 for pred in predictions]

    # Save classification report
    label_names = ['Cat', 'Dog']
    report = classification_report(y_test,
                                predictions,
                                target_names=label_names) 

    txtpath = os.path.join("..", "out", f"model_{model_name}_clf_report.txt")
    with open(txtpath, "w") as file:
        file.write(report) 

if __name__ == "__main__":
    args = input_parse()
    main(args.style, args.flip_shift)
