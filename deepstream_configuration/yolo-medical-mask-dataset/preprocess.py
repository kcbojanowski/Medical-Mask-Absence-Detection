import os
import glob
from PIL import Image

def resize_image(image_path, output_size):
    with Image.open(image_path) as img:
        img = img.resize(output_size, Image.Resampling.LANCZOS)
        img.save(image_path)

def resize_images(directory, pattern, size):
    for filename in glob.glob(os.path.join(directory, pattern)):
        resize_image(filename, size)
        print(f"Resized {filename}")

def rename_files(directory, pattern, extension):
    for count, filename in enumerate(sorted(glob.glob(os.path.join(directory, pattern)))):
        new_name = f"image_{count:05d}{extension}"
        os.rename(filename, os.path.join(directory, new_name))
        print(f"Renamed {filename} to {new_name}")

def main():
    image_directory = "test/images/"
    label_directory = "test/labels/"
    target_size = (1280, 720)

    # Resize JPG files
    #resize_images(image_directory, "*.jpg", target_size)

    # Rename JPG files
    rename_files(image_directory, "*.jpg", ".jpg")

    # Rename TXT files
    rename_files(label_directory, "*.txt", ".txt")

if __name__ == "__main__":
    main()
