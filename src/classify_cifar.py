print("[INFO]: Importing packages")
# general tools
import os
import numpy as np
import argparse
import tensorflow as tf

# data loading
from utils import (load_style_img, crop_resize, build_model, plot_imgs, plot_history)
from tensorflow.keras.datasets import cifar10

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
    print("[INFO]: Loading cifar10")
    ((X_train, y_train), (X_test, y_test)) = cifar10.load_data()

    # Subset cats and dogs
    X_train = X_train[np.isin(y_train, [3,5]).flatten()]
    y_train = y_train[np.isin(y_train, [3,5]).flatten()]
    X_test = X_test[np.isin(y_test, [3,5]).flatten()]
    y_test = y_test[np.isin(y_test, [3,5]).flatten()]

    # Normalize
    X_train = X_train.astype(np.float32) / 255.
    X_test = X_test.astype(np.float32) / 255.

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

        # Generate stylized images
        print("[INFO]: Generating stylized images")
        outputs = hub_module(X_train_t, style_img)

        # Convert stylized images back to original X_train format
        stylized_X_train = np.array(outputs[0])

        # Combine
        X_train = np.concatenate((X_train, stylized_X_train))
        y_train = np.concatenate((y_train, y_train))

        # Save figure of style transfer examples
        style_trim = style.replace(".png","")
        plot_imgs(images=[X_train[0], np.array(style_img[0]), X_train[10000]],
                  fname=f"cifar_{style_trim}_cat.png")
        plot_imgs(images=[X_train[7001], np.array(style_img[0]), X_train[17001]],
                  fname=f"cifar_{style_trim}_dog.png")

    # Make labels into 0s (cat) and 1s (dog)
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    
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
    model = build_model(cifar=True)

    # Define appropriate batch sizes
    if style != None:
        batch_size_train = 64
        batch_size_val = 16
    else:
        batch_size_train = 32
        batch_size_val = 8 
        
    # Fit model
    print("[INFO]: Model training")
    H = model.fit(train_generator.flow(X_train, y_train, batch_size=batch_size_train, subset='training'), 
            validation_data = train_generator.flow(X_train, y_train, batch_size=batch_size_val, subset = "validation"),
            epochs=10,
            verbose=1)
    
    # Save model and history plot
    if style != None:
        model_name = f"cifar_style_{style_trim}"
        plot_name = f"history_cifar_style_{style}"
    elif flip_shift:
        model_name = "cifar_flip_shift"
        plot_name = "history_cifar_flip_shift.png"
    else: 
        model_name = "cifar_no_aug"
        plot_name = "history_cifar_no_aug.png"
    
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