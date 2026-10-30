
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path

output_dir = Path("samples")
output_dir.mkdir(exist_ok=True)

# 1. Normal image
normal = Image.new("RGB", (640, 480), (120, 160, 190))
draw = ImageDraw.Draw(normal)
draw.rectangle((150, 100, 490, 380), fill=(220, 180, 130))
draw.text((220, 220), "NORMAL IMAGE", fill="black")
normal.save(output_dir / "normal.jpg")

# 2. Dark image
dark = normal.point(lambda pixel: int(pixel * 0.12))
dark.save(output_dir / "dark.jpg")

# 3. Blurry image
blurry = normal.filter(ImageFilter.GaussianBlur(radius=12))
blurry.save(output_dir / "blurry.jpg")

# 4. Bright image
bright = Image.new("RGB", (640, 480), (255, 255, 255))
draw = ImageDraw.Draw(bright)
draw.ellipse((180, 100, 460, 380), fill=(255, 255, 220))
bright.save(output_dir / "bright.jpg")

print("Four sample images created successfully!")
print("Location:", output_dir.resolve())
