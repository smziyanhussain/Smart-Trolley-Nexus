import os
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

output_dir = Path("dataset/smart_watch")
output_dir.mkdir(parents=True, exist_ok=True)

for i in range(100):
    img = Image.new('RGB', (224, 224), color=(np.random.randint(200), np.random.randint(200), np.random.randint(200)))
    draw = ImageDraw.Draw(img)

    draw.rectangle([70, 50, 150, 170], outline="black", width=4)
    draw.ellipse([95, 95, 125, 125], fill="black")

    img.save(output_dir / f"smart_watch_{i+1:03d}.jpg")

output_dir.exists(), len(list(output_dir.iterdir()))
