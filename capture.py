import cv2
import os

# === Ask for Product Name ===
class_name = input("📝 Enter product name: ").strip().lower().replace(" ", "_")
save_dir = f"dataset/{class_name}"
os.makedirs(save_dir, exist_ok=True)

# === Initialize Camera ===
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("❌ Camera not found.")
    exit()

img_count = len(os.listdir(save_dir))  # continue count if folder not empty
print("📸 Press SPACE to capture, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Could not read frame.")
        continue

    cv2.imshow("Manual Capture - Press SPACE to save", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # SPACE key to capture
        img_path = os.path.join(save_dir, f"{class_name}_{img_count:03d}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"✅ Saved: {img_path}")
        img_count += 1

    elif key == ord('q'):  # Quit
        print("🛑 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
