# convert_to_tflite.py

import tensorflow as tf

# Load the .h5 Keras model
model = tf.keras.models.load_model("product_recognizer_smallset.h5")

# Create TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: Reduce model size using float16 (recommended for Pi)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

# Convert
tflite_model = converter.convert()

# Save to .tflite file
with open("product_recognizer.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Conversion complete: product_recognizer.tflite created.")
