import os
from PIL import Image
import numpy as np
from moviepy.editor import ImageSequenceClip, AudioFileClip, concatenate_videoclips
from gtts import gTTS
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from pdf2image import convert_from_path


def images_from_folder(folder_path):
    image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
                   file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return sorted(image_files)  # Sort to ensure order


def describe_images(image_files):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    descriptions = []
    for image_file in image_files:
        image = Image.open(image_file).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=50)  # Set max_new_tokens
        description = processor.decode(out[0], skip_special_tokens=True)
        descriptions.append(description)

    return descriptions


def text_to_speech(descriptions, output_dir="audio_files"):
    os.makedirs(output_dir, exist_ok=True)
    audio_files = []
    for i, description in enumerate(descriptions):
        tts = gTTS(description, lang='en')
        audio_path = os.path.join(output_dir, f"slide_{i + 1}.mp3")
        tts.save(audio_path)
        audio_files.append(audio_path)

    return audio_files


def create_video(image_files, audio_files, output_file, fps=24):
    clips = []
    for image_file, audio_file in zip(image_files, audio_files):
        audio = AudioFileClip(audio_file)
        duration = audio.duration
        img = Image.open(image_file)
        img_array = np.array(img)
        img_clip = ImageSequenceClip([img_array], durations=[duration])
        img_clip = img_clip.set_audio(audio)
        img_clip = img_clip.set_fps(fps)  # Set the fps for each image clip
        clips.append(img_clip)

    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(output_file, codec="libx264", audio_codec="aac",
                               fps=fps)  # Set the fps for the final video


def convert_pdf_to_images(pdf_path, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Convert PDF to images
    images = convert_from_path(pdf_path)
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f"page_{i + 1}.png")
        image.save(image_path, 'PNG')

    return output_folder


if __name__ == "__main__":
    pdf_path = "/Users/vinhvu/BKInnovationSlide2Vid/HCMUT-TIMETABLE/Chapter_0_Complex_numbers.pdf"  # Path to your PDF file
    output_folder = "/Users/vinhvu/BKInnovationSlide2Vid/HCMUT-TIMETABLE/image"  # Path to save the images
    output_file = "output_demo.mp4"

    convert_pdf_to_images(pdf_path, output_folder)
    image_files = images_from_folder(output_folder)

    descriptions = describe_images(image_files)
    print(descriptions)

    audio_files = text_to_speech(descriptions)

    create_video(image_files, audio_files, output_file)
