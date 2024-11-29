import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load your pre-trained captioning model (placeholder)
# Replace with your actual captioning model
def load_caption_model():
    # Placeholder for loading the model
    # model = tf.keras.models.load_model('path_to_captioning_model.h5')
    # Use your model instead
    pass

# Generate a caption for the given image
def generate_caption(image):
    # Placeholder logic for generating caption using your model
    return "Women Cooking Food."

# Load an image, process it, and display the caption
def load_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    # Display the image in the interface
    img = Image.open(file_path)
    img = img.resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

    # Process the image
    img_array = img_to_array(img.resize((224, 224)))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Generate a caption
    caption = generate_caption(img_array)
    caption_label.config(text=f"Caption: {caption}")

# Initialize the GUI
root = tk.Tk()
root.title("Image Captioning")
root.geometry("400x500")

# Image display
image_label = tk.Label(root)
image_label.pack(pady=20)

# Load Image Button
load_button = tk.Button(root, text="Load Image", command=load_image)
load_button.pack(pady=10)

# Caption display
caption_label = tk.Label(root, text="Caption: ", font=("Helvetica", 14))
caption_label.pack(pady=10)

# Run the GUI
root.mainloop()
