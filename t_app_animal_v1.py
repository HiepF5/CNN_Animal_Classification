import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

model = tf.keras.models.load_model('model_3.h5')

class_labels = ['cat', 'chicken', 'cow', 'crab', 'dog', 'elephant', 'horse', 'whale']

app = tk.Tk()
app.title("Animal Classifier")

last_directory = "/"

screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()

content_width = 450
content_height = 450

x = (screen_width - content_width) // 2
y = (screen_height - content_height) // 2

app.geometry(f"{content_width}x{content_height}+{x}+{y}")

def open_file_dialog():
    global last_directory
    file_path = filedialog.askopenfilename(initialdir=last_directory, title="Select Image", 
                                           filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        last_directory = os.path.dirname(file_path)
        display_image(file_path)

def display_image(file_path):
    global img
    image = Image.open(file_path)
    image = image.resize((200, 200))  
    img = ImageTk.PhotoImage(image)
    image_label.config(image=img)
    image_label.image = img
    predict_button["state"] = "normal"
    selected_image_path.set(file_path)

def predict_image():
    image_path = selected_image_path.get()
    img = image.load_img(image_path, target_size=(200, 200))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  

    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]

    result_label.config(text=f"Predicted Class: {predicted_class}")

title_label = tk.Label(app, text="Animals Classifier", font=("Helvetica", 20, "bold"), bg='#f0f0f0')  
title_label.pack(side=tk.TOP, pady=15)

image_label = tk.Label(app)
image_label.pack()

selected_image_path = tk.StringVar()

result_label = tk.Label(app, text="", font=("Helvetica", 12), bg='#f0f0f0')  
result_label.pack(pady=15)

button_frame = tk.Frame(app, bg='#f0f0f0')
button_frame.pack(side=tk.BOTTOM, pady=15)

open_button = tk.Button(button_frame, text="Open Image", command=open_file_dialog,
                        width=20, height=2, bg="#4CAF50", fg="white", activebackground="#45a049")  
open_button.pack(side=tk.LEFT, padx=5)

predict_button = tk.Button(button_frame, text="Predict", command=predict_image, state="disabled",
                           width=20, height=2, bg="#45a050", fg="white", activebackground="#45a049")  
predict_button.pack(side=tk.LEFT, padx=5)

app.mainloop()
