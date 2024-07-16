from django.shortcuts import render
from PIL import Image, ImageDraw
from io import BytesIO
from transformers import pipeline
import cv2
import numpy as np
import base64

# Create pipelines for image and audio classification
image_classification_pipeline = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")
audio_classification_pipeline = pipeline(
    "audio-classification",
    model="HyperMoon/wav2vec2-base-960h-finetuned-deepfake",
    ffmpeg_path="C:/ffmpeg"  # Update this path to your ffmpeg location
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
                        deepfake_score = deepfake_results[0]['score'] * 100
                        return render(request, 'deepfake_detection.html', {'results': deepfake_results, 'image_with_boxes': image_with_boxes, 'score': deepfake_score})
                elif file_type == 'audio':
                    deepfake_results = detect_deepfake_audio(uploaded_file)
                    return render(request, 'deepfake_detection.html', {'audio_results': deepfake_results})
                elif file_type == 'video':
                    deepfake_results, deepfake_percentage, error_message = detect_deepfake_video(uploaded_file)
                    if error_message:
                        return render(request, 'deepfake_detection.html', {'error': error_message})
                    else:
                        return render(request, 'deepfake_detection.html', {'video_results': deepfake_results, 'deepfake_percentage': deepfake_percentage})
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

    image = Image.open(image_file)

    if image.mode == "RGBA":
        image = image.convert("RGB")

    results = model(image)

    for result in results:
        if result['label'] == 'FAKE':
            print("Bounding Box Coordinates:", result['box'])

    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)
    for result in results:
        if result['label'] == 'FAKE':
            box = result['box']
            draw.rectangle(box, outline="red")

    image_bytes = BytesIO()
    image_with_boxes.save(image_bytes, format='JPEG')
    image_with_boxes_base64 = base64.b64encode(image_bytes.getvalue()).decode('utf-8')

    return results, image_with_boxes_base64, None

def detect_deepfake_audio(audio_file):
    audio_data = audio_file.read()
    results = audio_classification_pipeline(audio_data)

    formatted_results = []
    for result in results:
        label = result['label']
        score = result['score']
        formatted_score = "{:.2%}".format(score) if score is not None else "N/A"
        formatted_results.append({'label': label, 'score': formatted_score})

    for formatted_result in formatted_results:
        print(f"Label: {formatted_result['label']}, Score: {formatted_result['score']}")

    return formatted_results

def detect_deepfake_video(video_file):
    cap = cv2.VideoCapture(video_file.temporary_file_path())

    if not cap.isOpened():
        return None, None, 'Unable to open video file'

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = max(1, frame_count // 30)

    fake_scores = []

    for i in range(0, frame_count, frame_skip):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break

        frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = image_classification_pipeline(frame_image)
        for result in results:
            if result['label'] == 'FAKE':
                fake_scores.append(result['score'])

    cap.release()

    if not fake_scores:
        return {'average_fake_score': 0, 'frame_count': 0}, "0.00%", None

    average_fake_score = (sum(fake_scores) / len(fake_scores)) * 100
    deepfake_percentage = "{:.2f}%".format(average_fake_score)

    return {'average_fake_score': average_fake_score, 'frame_count': len(fake_scores)}, deepfake_percentage, None
