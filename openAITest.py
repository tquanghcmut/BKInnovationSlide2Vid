import os
import base64
import requests
from PIL import Image
import numpy as np
from moviepy.editor import ImageSequenceClip, AudioFileClip, concatenate_videoclips
from pathlib import Path
import openai
from pathlib import Path
from openai import OpenAI

# OpenAI API Key
api_key = ""  # Thay 'YOUR_OPENAI_API_KEY' bằng API key của bạn

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# Bước 1: Đọc tất cả các ảnh từ thư mục
def images_from_folder(folder_path):
    image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
                   file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return sorted(image_files)  # Sắp xếp để đảm bảo thứ tự

# Bước 2: Sử dụng GPT-4 để mô tả từng ảnh
def describe_single_image(image_files):
    descriptions = []
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    for image_file in image_files:
        base64_image = encode_image(image_file)
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are the professor, and this is one slide of all presentation. Give a small lecture of  this slide like a professor. Remember, just this slide, aware of length, you should focus on important part and inform shortly,enough for a slide. You should give me raw text. Don't give me outline number. This support overall lecture CNN and RNN for sentiment analysis.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            try:
                description = response.json()['choices'][0]['message']['content']
                descriptions.append(description)
            except KeyError:
                print(f"Error in response for image {image_file}: {response.json()}")
                descriptions.append("Error in generating description.")
        else:
            print(f"Failed to get response for image {image_file}: {response.text}")
            descriptions.append("Failed to get description.")
    return descriptions


client = openai.OpenAI(api_key=api_key)

# Bước 3: Chuyển đổi văn bản mô tả thành giọng nói với TTS của OpenAI
def text_to_speech_with_openai(descriptions, output_dir="audio_files"):
    os.makedirs(output_dir, exist_ok=True)
    audio_files = []
    openai.api_key = api_key

    for i, description in enumerate(descriptions):
        speech_file_path = Path(output_dir) / f"slide_{i + 1}.mp3"
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=description
        )
        with open(speech_file_path, "wb") as f:
            f.write(response.content)
        audio_files.append(str(speech_file_path))
    return audio_files

# Bước 4: Tạo video từ các ảnh và âm thanh tương ứng
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

    if clips:
        final_clip = concatenate_videoclips(clips, method="compose")
        final_clip.write_videofile(output_file, codec="libx264", audio_codec="aac", fps=fps)  # Set the fps for the final video
    else:
        print("No clips to concatenate")

if __name__ == "__main__":
    folder_path = "/Users/twang/PycharmProjects/HCMUT-TIMETABLE/demo_image_lect3"  # Đường dẫn đến thư mục chứa ảnh
    output_file = "lect3.mp4"

    # Bước 1: Lấy ảnh từ thư mục
    image_files = images_from_folder(folder_path)

    # Bước 2: Mô tả từng ảnh
    descriptions = describe_single_image(image_files)
    print("Descriptions:", descriptions)

    # Bước 3: Chuyển đổi mô tả thành giọng nói với OpenAI TTS
    audio_files = text_to_speech_with_openai(descriptions)

    # Bước 4: Tạo video từ ảnh và âm thanh tương ứng
    create_video(image_files, audio_files, output_file)
