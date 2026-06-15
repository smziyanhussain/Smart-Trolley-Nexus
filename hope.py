import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import time
import csv
import winsound

# ------------------ Load Model and Dataset ------------------
model = load_model('product_recognizer_smallset.h5')
dataset_path = 'dataset/'
class_names = sorted(os.listdir(dataset_path))

# ------------------ Load Prices ------------------
prices = {}
with open('prices.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        prices[row['product']] = int(row['price'])

# ------------------ Cart & Variables ------------------
cart = []
total = 0
front_threshold = 70
back_threshold = 70
last_detection_time = time.time()
last_detected = None
startup_time = time.time()
warmup_frames = 15
frame_count = 0
previous_predictions = []
prediction_window = 5
detected_classes = set()
theft_popup_active = False

# ------------------ Predict Function ------------------
def predict_from_frame(frame):
    img = cv2.resize(frame, (224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    predictions = model.predict(x, verbose=0)[0]

    for idx, class_name in enumerate(class_names):
        if class_name in detected_classes:
            predictions[idx] = 0.0

    max_index = np.argmax(predictions)
    predicted_class = class_names[max_index]
    confidence = predictions[max_index] * 100
    return predicted_class, confidence

# ------------------ GUI Setup ------------------
root = tk.Tk()
root.title("🔝 Dual Camera Smart Checkout")
root.geometry("1000x700")
root.configure(bg="black")

left_frame = tk.Frame(root, width=450, height=700, bg="black")
left_frame.pack(side="left", fill="both", expand=False)

tk.Label(left_frame, text="Front View", font=("Arial", 12), fg="white", bg="black").pack()
front_cam = tk.Label(left_frame, bg="black")
front_cam.pack(padx=10, pady=5)

tk.Label(left_frame, text="Back View", font=("Arial", 12), fg="white", bg="black").pack()
back_cam = tk.Label(left_frame, bg="black")
back_cam.pack(padx=10, pady=5)

right_frame = tk.Frame(root, width=550, height=700, bg="#1e1e1e")
right_frame.pack(side="right", fill="both", expand=True)

tk.Label(right_frame, text="🗿 Bill Summary", font=("Arial", 16, "bold"), bg="#1e1e1e", fg="white").pack(pady=10)

bill_listbox = tk.Listbox(right_frame, width=45, height=15, font=("Arial", 12), bg="black", fg="white", bd=0)
bill_listbox.pack(padx=20)

total_var = tk.StringVar(value="Total: Rs. 0")
tk.Label(right_frame, textvariable=total_var, font=("Arial", 14, "bold"), bg="#1e1e1e", fg="lime").pack(pady=10)

message_var = tk.StringVar()
tk.Label(right_frame, textvariable=message_var, font=("Arial", 12), bg="#1e1e1e", fg="orange").pack(pady=5)

# ------------------ Camera Setup ------------------
cap1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap2 = cv2.VideoCapture(2, cv2.CAP_DSHOW)


# Set both to 720p resolution
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: One or both cameras are not accessible.")
    exit()

webcam_running = True

# ------------------ Frame Update ------------------
def update_frame():
    global last_detected, last_detection_time, frame_count, total, previous_predictions, theft_popup_active

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        front_cam.after(100, update_frame)
        return

    for label, frame, cam_label in [("Front", frame1, front_cam), ("Back", frame2, back_cam)]:
        disp = cv2.resize(frame, (420, 240))
        disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        cam_label.imgtk = ImageTk.PhotoImage(Image.fromarray(disp))
        cam_label.configure(image=cam_label.imgtk)

    frame_count += 1
    if frame_count <= warmup_frames:
        front_cam.after(30, update_frame)
        return

    if frame_count % 3 != 0:
        front_cam.after(30, update_frame)
        return

    if webcam_running and time.time() - startup_time > 2:
        front_class, front_conf = predict_from_frame(frame1)
        back_class, back_conf = predict_from_frame(frame2)
        current_time = time.time()

        print(f"Front: {front_class} ({front_conf:.2f}%) | Back: {back_class} ({back_conf:.2f}%)")

        if (
            front_conf >= front_threshold and back_conf >= back_threshold and
            front_class != back_class and
            front_class != "no_product" and back_class != "no_product"
        ):
            if not theft_popup_active:
                theft_popup_active = True
                message_var.set("\u26a0\ufe0f Theft Alert: Mismatch Detected!")
                previous_predictions.clear()

                def show_theft_popup():
                    winsound.Beep(1000, 500)
                    popup = tk.Toplevel(root)
                    popup.title("\u26a0\ufe0f Theft Alert")
                    popup.geometry("300x150")
                    popup.configure(bg="black")
                    tk.Label(
                        popup,
                        text="\u26a0\ufe0f Product mismatch detected!\nPlease verify the trolley.",
                        font=("Arial", 12, "bold"), bg="black", fg="red"
                    ).pack(pady=20)
                    def close_popup():
                        popup.destroy()
                        message_var.set("")
                        global theft_popup_active
                        theft_popup_active = False
                    tk.Button(
                        popup, text="OK", font=("Arial", 11), command=close_popup,
                        bg="red", fg="white", padx=10, pady=5
                    ).pack()
                show_theft_popup()
            front_cam.after(30, update_frame)
            return

        new_product = None
        if front_class != "no_product" and front_conf >= front_threshold:
            new_product = front_class
        elif back_class != "no_product" and back_conf >= back_threshold:
            new_product = back_class

        if new_product:
            previous_predictions.append(new_product)
            if len(previous_predictions) > prediction_window:
                previous_predictions.pop(0)
            most_common = max(set(previous_predictions), key=previous_predictions.count)
            count = previous_predictions.count(most_common)
            if most_common == last_detected:
                duration = current_time - last_detection_time
            else:
                last_detection_time = current_time
                duration = 0
                last_detected = most_common
            if count >= 4 and duration >= 0 and most_common not in detected_classes:
                price = prices.get(most_common, 0)
                cart.append((most_common, price))
                detected_classes.add(most_common)
                bill_listbox.insert(tk.END, f"{most_common:10} - Rs. {price}")
                total += price
                total_var.set(f"Total: Rs. {total}")
                message_var.set(f"✔ {most_common} added to bill")
                previous_predictions.clear()
                last_detected = None
                root.after(3000, lambda: message_var.set(""))

    front_cam.after(30, update_frame)

# ------------------ Checkout Function ------------------
def checkout():
    global webcam_running
    webcam_running = False
    cap1.release()
    cap2.release()

    popup = tk.Toplevel(root)
    popup.title("🗳 Final Bill")
    popup.geometry("350x400")
    popup.configure(bg="#121212")

    tk.Label(popup, text="📍 Smart Shopping Receipt", font=("Arial", 16, "bold"), bg="#121212", fg="white").pack(pady=10)

    receipt_text = tk.Text(popup, font=("Courier", 12), bg="#1e1e1e", fg="white", bd=0)
    receipt_text.pack(padx=10, pady=10, fill="both", expand=True)
    receipt_text.insert(tk.END, "Item         Price\n")
    receipt_text.insert(tk.END, "--------------------\n")
    for item, price in cart:
        receipt_text.insert(tk.END, f"{item:12} Rs. {price}\n")
    receipt_text.insert(tk.END, "--------------------\n")
    receipt_text.insert(tk.END, f"TOTAL        Rs. {total}\n")
    receipt_text.config(state="disabled")

# ------------------ Checkout Button ------------------
btn_frame = tk.Frame(right_frame, bg="#1e1e1e")
btn_frame.pack(pady=20)

def on_enter(e):
    checkout_btn['background'] = '#28a745'
    checkout_btn['foreground'] = 'white'

def on_leave(e):
    checkout_btn['background'] = '#218838'
    checkout_btn['foreground'] = 'white'

checkout_btn = tk.Button(
    btn_frame,
    text="✅ Checkout",
    font=("Arial", 12, "bold"),
    bg="#218838", fg="white",
    activebackground="#28a745",
    activeforeground="white",
    command=checkout,
    relief="flat", padx=10, pady=5
)
checkout_btn.pack()
checkout_btn.bind("<Enter>", on_enter)
checkout_btn.bind("<Leave>", on_leave)

# ------------------ Start ------------------
update_frame()

def on_close():
    cap1.release()
    cap2.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
