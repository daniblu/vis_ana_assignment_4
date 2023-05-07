# IMPORTING PACKAGES
print("[INFO]: Importing packages")
# general tools
import os
import numpy as np
import argparse
import tensorflow as tf

# data loading
from utils import (st_load, crop_center, plot_history)
from tensorflow.keras.datasets import cifar10

# image processsing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# style transfer model
import tensorflow_hub as hub

# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 VGG16)

# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     BatchNormalization)

# generic model object
from tensorflow.keras.models import Model

# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

#scikit-learn
from sklearn.preprocessing import LabelBinarizer
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
        style_img = st_load(style_path)

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

        # SAVE FIGURE OF STYLE TRANSFER EXAMPLES

    # One-hot encode labels
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    
    # Define image data generators
    if flip_shift:
        train_generator = ImageDataGenerator(horizontal_flip=True,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            preprocessing_function=preprocess_input, 
                                            validation_split=0.2,
                                            seed=5)
        train_generator.fit(X_train)
    else:
        train_generator = ImageDataGenerator(preprocessing_function=preprocess_input,
                                            validation_split=0.2,
                                            seed=5)
        train_generator.fit(X_train)
    
    test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator.fit(X_test)

    # Load model without classifier layers
    model = VGG16(include_top=False, 
                pooling='avg',
                input_shape=(32, 32, 3))

    # Mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
        
    # Add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    bn = BatchNormalization()(flat1)
    class1 = Dense(256, 
                activation='relu')(bn)
    class2 = Dense(128, 
                activation='relu')(class1)
    output = Dense(1, 
                activation='sigmoid')(class2)

    # Define new model
    model = Model(inputs=model.inputs, 
                outputs=output)

    # Define optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.9)
    sgd = SGD(learning_rate=lr_schedule)

    # Compile
    model.compile(optimizer=sgd,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Define appropriate batch sizes
    if flip_shift:
        batch_size_train = 32
        batch_size_val = 8
    else: 
        batch_size_train = 64
        batch_size_val = 16

    # Fit model
    print("[INFO]: Model training")
    H = model.fit(train_generator.flow(X_train, y_train, batch_size=batch_size_train), 
            validation_data = train_generator.flow(X_train, y_train, batch_size=batch_size_val, subset = "validation"),
            epochs=8,
            verbose=1)
    
    # Save model and history plot
    if style != None:
        style_trim = style.replace(".png","")
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
    
    plot_history(H, epochs=8, title=plot_name)

    # Predict
    print("[INFO]: Predicting")
    predictions = model.predict(test_generator.flow(X_test,
                                                    batch_size=1,
                                                    shuffle=False))

    # Save classification report
    label_names = ['Cat', 'Dog']
    report = classification_report(y_test.flatten(),
                        predictions.flatten(),
                        target_names=label_names) 

    txtpath = os.path.join("..", "reports", f"model_{model_name}_classification_report.txt")
    with open(txtpath, "w") as file:
        file.write(report) 

if __name__ == "__main__":
    args = input_parse()
    main(args.style, args.flip_shift)