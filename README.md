# Deepfake Detection

## Overview 🔭
In today's digital world, identifying deepfake content is crucial for maintaining the integrity of visual and audio media. Our tool aims to streamline this process by automating the detection of deepfakes in images, audio, and video files, providing accurate and efficient results.

## Features⭐️

1. **Comprehensive Deepfake Detection🔍**

   Our tool leverages pre-trained models from Hugging Face to detect deepfakes in various media formats. By analyzing content using these models, we provide users with a thorough assessment of potential deepfake elements.

2. **Transformer Model Integration🤗**

   To enhance the accuracy of deepfake detection, we've integrated Transformer models trained on labeled deepfake data. These models predict the likelihood of content being a deepfake, helping prioritize and address suspicious media.

3. **User-Friendly Interface😀**

   Our web application features an intuitive and user-friendly interface, built using Python Django for the backend and HTML/CSS for the frontend. This makes it easy for users to upload their files, initiate analysis, and interpret results. Visualizations and highlighting of deepfake elements aid in understanding and addressing issues.

## Getting Started 🚀

To use our tool, follow these steps:

1. **Upload File:** Navigate to our web application and upload your image, audio, or video file.
2. **Initiate Analysis:** Start the deepfake detection process to identify potential deepfakes and predict their likelihood.
3. **Review Results:** Explore the analysis results, including highlighted deepfake elements and predicted scores.

## Installation 🔨🔧

Our tool is accessible through a web interface, making it easy to use without the need for local installation.

## Requirements 🔧
Web browser with JavaScript enabled Stable internet connection for accessing the web application

## Technologies Used 💻
**Backend:** Python Django

**Frontend:** HTML, CSS,Javascript

**Machine Learning Models:** 
Image: dima806/deepfake_vs_real_image_detection
Audio: HyperMoon/wav2vec2-base-960h-finetuned-deepfake

**Database:** PostgreSQL

**Hosting:** Azure App Services

## Detailed Project Description 📜

Our Deepfake Detection project is designed to detect manipulated media files using advanced machine learning models. The tool supports detection in images, audio, and video files by leveraging state-of-the-art Transformer models available on Hugging Face.

### Image Deepfake Detection

For image analysis, we use the `dima806/deepfake_vs_real_image_detection` model. This model helps in identifying whether an image has been manipulated. The process involves:
- Uploading an image file.
- Running the image through the deepfake detection model.
- Displaying the results with a probability score indicating the likelihood of the image being a deepfake.
- Highlighting detected deepfake elements within the image.

### Audio Deepfake Detection

For audio analysis, we use the `HyperMoon/wav2vec2-base-960h-finetuned-deepfake` model. This model detects deepfake audio by:
- Uploading an audio file (MP3 format).
- Processing the audio through the model.
- Displaying results with a score indicating the probability of the audio being manipulated.

### Video Deepfake Detection

Video analysis is handled using a combination of image frames extracted from the video. The process includes:
- Uploading a video file (MP4 format).
- Extracting frames from the video at regular intervals.
- Running each frame through the deepfake detection model.
- Aggregating results to provide an overall deepfake score for the video.
- Highlighting frames with detected deepfake elements.
