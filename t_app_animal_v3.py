import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from annoy import AnnoyIndex  # Cần cài đặt thư viện Annoy

# Tải mô hình và khởi tạo biến toàn cục
model = tf.keras.models.load_model('model_1.h5')
class_labels = ['cat', 'chicken', 'cow', 'crab', 'dog', 'elephant', 'horse', 'whale']
database_features = []
database_paths = []

# Tạo và cấu hình ứng dụng Tkinter
app = tk.Tk()
app.title("Animal Classifier")

screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()
content_width = 450
content_height = 450
x = (screen_width - content_width) // 2
y = (screen_height - content_height) // 2
app.geometry(f"{content_width}x{content_height}+{x}+{y}")

# Hàm mở hộp thoại chọn tệp và hiển thị hình ảnh
def open_file_dialog():
    global last_directory
    file_path = filedialog.askopenfilename(initialdir=last_directory, title="Select Image", 
                                           filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        last_directory = os.path.dirname(file_path)
        display_image(file_path)

# Hàm hiển thị hình ảnh đã chọn
def display_image(file_path):
    global img
    image = Image.open(file_path)
    image = image.resize((200, 200))  
    img = ImageTk.PhotoImage(image)
    image_label.config(image=img)
    image_label.image = img
    predict_button["state"] = "normal"
    selected_image_path.set(file_path)

# Hàm trích xuất đặc trưng từ hình ảnh
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(200, 200))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  

    return model.predict(img_array)

# Hàm tính toán độ tương đồng giữa hai đặc trưng
def calculate_similarity(feature1, feature2):
    return np.dot(feature1, feature2.T) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))

# Hàm tìm và hiển thị top k hình ảnh tương đồng
def display_similar_images(selected_image_features, top_k=3):
    index = AnnoyIndex(len(selected_image_features), 'euclidean')
    for i, feature in enumerate(database_features):
        index.add_item(i, feature)

    index.build(10)  # Số lượng cây trong forest

    similarities = index.get_nns_by_vector(selected_image_features, top_k, include_distances=True)
    for i, (index, similarity) in enumerate(zip(similarities[0], similarities[1])):
        image_path = database_paths[index]
        similarity = 1 / (1 + similarity)  # Chuyển đổi khoảng cách sang độ tương đồng (giá trị càng lớn, càng tương đồng)
        display_similar_image(image_path, similarity)

# Hàm hiển thị hình ảnh tương đồng
def display_similar_image(image_path, similarity):
    similarity_scalar = similarity.item()
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

# Hàm dự đoán lớp của hình ảnh và tìm hình ảnh tương đồng
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
    display_similar_images(selected_image_features)

# Cấu hình giao diện người dùng
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
                           width=20, height=2,bg="#45a050", fg="white", activebackground="#45a049")  
predict_button.pack(side=tk.LEFT, padx=5)

# Tải các đặc trưng và đường dẫn của tất cả các hình ảnh trong cơ sở dữ liệu
def load_image_features(dataset_path):
    global database_features, database_paths
    for root, dirs, files in os.walk(dataset_path):
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(root, filename)
                feature = extract_features(image_path)
                database_paths.append(image_path)
                database_features.append(feature)

# Mở cửa sổ để chọn thư mục chứa cơ sở dữ liệu và tải các đặc trưng của hình ảnh
def load_database():
    global database_features, database_paths
    dataset_path = filedialog.askdirectory(title="Select Database Folder")
    if dataset_path:
        load_image_features(dataset_path)
        predict_button["state"] = "normal"

# Thêm nút để tải cơ sở dữ liệu
load_button = tk.Button(button_frame, text="Load Database", command=load_database,
                        width=20, height=2, bg="#007bff", fg="white", activebackground="#0056b3")  
load_button.pack(side=tk.LEFT, padx=5)

app.mainloop()
