from django.shortcuts import render
from PIL import Image, ImageDraw
from io import BytesIO
from transformers import pipeline
import cv2
import numpy as np
import base64
# Create a pipeline for audio classification
ffmpeg_path = "C:\ffmpeg"
audio_classification_pipeline = pipeline(
    "audio-classification",
    model="HyperMoon/wav2vec2-base-960h-finetuned-deepfake",
    ffmpeg_path=ffmpeg_path
)

def deepfake_detection(request):
    if request.method == 'POST':
        if 'file' in request.FILES:
            uploaded_file = request.FILES['file']
            file_type = uploaded_file.content_type.split('/')[0]  # Check file type (image, audio, video)

            try:
                if file_type == 'image':
                    deepfake_results, image_with_boxes, error_message = detect_deepfake_image(uploaded_file)
                    if error_message:
                        return render(request, 'deepfake_detection.html', {'error': error_message})
                    else:
                        # Calculate percentage
                        deepfake_score = deepfake_results[0]['score'] * 100
                        return render(request, 'deepfake_detection.html', {'results': deepfake_results, 'image_with_boxes': image_with_boxes, 'score': deepfake_score})
                elif file_type == 'audio':
                    deepfake_results = detect_deepfake_audio(uploaded_file)
                    return render(request, 'deepfake_detection.html', {'audio_results': deepfake_results})
                elif file_type == 'video':
                    deepfake_results, video_with_boxes, error_message = detect_deepfake_video(uploaded_file)
                    if error_message:
                        return render(request, 'deepfake_detection.html', {'error': error_message})
                    else:
                        return render(request, 'deepfake_detection.html', {'results': deepfake_results, 'video_with_boxes': video_with_boxes})
                else:
                    return render(request, 'deepfake_detection.html', {'error': 'Unsupported file type'})

            except Exception as e:
                return render(request, 'deepfake_detection.html', {'error': f'Error occurred: {str(e)}'})
        else:
            return render(request, 'deepfake_detection.html', {'error': 'No file uploaded'})
    else:
        return render(request, 'deepfake_detection.html')



def detect_deepfake_image(image_file):
    model = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")

    # Load image
    image = Image.open(image_file)

    # Convert image to RGB mode if it's in RGBA mode (with transparency)
    if image.mode == "RGBA":
        image = image.convert("RGB")

    # Perform deepfake detection
    results = model(image)

    # Print out the coordinates of the bounding boxes
    for result in results:
        if result['label'] == 'FAKE':
            print("Bounding Box Coordinates:", result['box'])

    # Create a copy of the original image to draw bounding boxes on
    image_with_boxes = image.copy()

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image_with_boxes)
    for result in results:
        if result['label'] == 'FAKE':
            box = result['box']
            draw.rectangle(box, outline="red")

    # Convert image with bounding boxes to bytes for rendering
    image_bytes = BytesIO()
    image_with_boxes.save(image_bytes, format='JPEG')
    image_with_boxes_base64 = base64.b64encode(image_bytes.getvalue()).decode('utf-8')

    return results, image_with_boxes_base64, None


def detect_deepfake_audio(audio_file):
    # Read the audio file as bytes
    audio_data = audio_file.read()

    # Use the audio classification pipeline to detect deepfake in audio
    results = audio_classification_pipeline(audio_data)

    # Format scores as percentages
    formatted_results = []
    for result in results:
        label = result['label']
        score = result['score']
        formatted_score = "{:.2%}".format(score) if score is not None else "N/A"  # Format score as percentage with 2 decimal places
        formatted_results.append({'label': label, 'score': formatted_score})

    # Print the formatted results
    for formatted_result in formatted_results:
        print(f"Label: {formatted_result['label']}, Score: {formatted_result['score']}")

    return formatted_results




def detect_deepfake_video(video_file):
    # Create a video classification pipeline
    video_classification_pipeline = pipeline("video-classification", model="muneeb1812/videomae-base-fake-video-classification")

    # Read the video file as bytes
    video_data = BytesIO(video_file.read())

    # Classify the video file
    results = video_classification_pipeline(video_data)

    # Process the classification results
    classified_frames = []
    for frame_results in results:
        classified_frames.extend(frame_results)

    # Print the classification results
    for result in classified_frames:
        print(f"Label: {result['label']}, Score: {result['score']}")

    return classified_frames