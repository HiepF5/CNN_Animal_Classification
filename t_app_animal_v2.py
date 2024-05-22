import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

model = tf.keras.models.load_model('model_3.h5')

class_labels =['cat', 'chicken', 'cow', 'crab', 'dog', 'elephant', 'horse', 'whale']

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

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(200, 200))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  

    return model.predict(img_array)

def calculate_similarity(feature1, feature2):
    return np.dot(feature1, feature2.T) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))

def get_top_similar_images(selected_image_features, database_features, database_paths, top_k=3):
    similarities = []
    for feature, path in zip(database_features, database_paths):
        similarity = calculate_similarity(selected_image_features, feature)
        similarities.append((path, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def display_similar_images(similar_images):
    # Create a new window to display similar images
    similar_window = tk.Toplevel(app)
    similar_window.title("Top 3 Similar Images")

    for i, (image_path, similarity) in enumerate(similar_images):
        similarity_scalar = similarity.item()  # Convert numpy array to scalar float
        image_frame = tk.Frame(similar_window)
        image_frame.pack(pady=5)

        label = tk.Label(image_frame, text=f"Similarity: {similarity_scalar:.2f}")
        label.pack()

        img = Image.open(image_path)
        img = img.resize((100, 100))
        img_photo = ImageTk.PhotoImage(img)

        img_label = tk.Label(image_frame, image=img_photo)
        img_label.image = img_photo
        img_label.pack(side=tk.LEFT, padx=5)

def predict_image():
    image_path = selected_image_path.get()
    img = image.load_img(image_path, target_size=(200, 200))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  

    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]

    result_label.config(text=f"Predicted Class: {predicted_class}")
    selected_image_features = extract_features(image_path)

    dataset_path = "Data/valid"
    database_paths = load_image_paths(dataset_path)
    database_features = [extract_features(img_path) for img_path in database_paths]

    top_similar_images = get_top_similar_images(selected_image_features, database_features, database_paths)

    display_similar_images(top_similar_images)

def load_image_paths(dataset_path):
    image_paths = []
    for root, dirs, files in os.walk(dataset_path):
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_paths.append(os.path.join(root, filename))
    return image_paths

title_label = tk.Label(app, text="Animal Classifier", font=("Helvetica", 20, "bold"), bg='#f0f0f0')  
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
