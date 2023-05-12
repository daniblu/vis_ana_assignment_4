# @title Define image loading and visualization functions  { display-mode: "form" }
import functools, os
from matplotlib import gridspec
import matplotlib.pylab as plt
import tensorflow as tf
import numpy as np

def crop_resize(image, image_size=(256, 256)):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  image = tf.image.resize(image, image_size, preserve_aspect_ratio=True)
  return image

@functools.lru_cache(maxsize=None)
def load_style_img(image_path):
  """Loads and preprocesses images."""
  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]
  if img.max() > 1.0:
    img = img / 255.
  if len(img.shape) == 3:
    img = tf.stack([img, img, img], axis=-1)
  # remove alpha-layer from .png
  if image_path[-4:] == '.png':
    img = img[:,:,:,:3]
  return img

def get_relpaths(sub="training_set", subsub="cats"):
  dir = os.path.join("..","in","dataset",sub,subsub)
  files = os.listdir(dir)
  relpaths = [os.path.relpath(os.path.join(dir, file)) for file in files]
   
  return relpaths

# image plotting function
def plot_imgs(images, fname):
    titles = ['content', 'style', 'stylized']

    plt.style.use('seaborn-white')

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(9,3))
    for i, ax in enumerate(axs.flatten()):
        plt.sca(ax)
        plt.imshow(images[i])
        plt.axis('off')
        plt.title(f"{titles[i]}")

    plt.tight_layout()
    plt.savefig(os.path.join("..","out",fname))

# history plotting function
def plot_history(H, epochs, title):
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    
    plt.savefig(os.path.join("..","models",f"{title}"))