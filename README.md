# AI-Model-Development-for-Counting-Fruit-Bags-Pear-Bags-
Develop vision AI Model and methodology for applyıng detection, tracking and deduplication for detect and count fruit bags

In pear orchards, fruit bags are used to improve pear quality. 
Every year, in collaboration with disaster insurance companies, the number of fruit bags is counted in spring and after natural disasters (e.g., typhoons) to assess the level of damage. 
The aim of this project is to develop a vision AI model that can  automatically count fruit bags from images and videos collected through cameras, eliminating the need for manual counting. 
For this purpose, i developed AI model for detection and applied tracking and deduplication methodology on video.

## Dataset

Dataset provided by company.Dataset consist of 4889 pearbags images and Provided images (videos) which contains fruit bags.
For applying tracking and deduplication methodology, i combined 4889 sequentially cropped images that we used camera dataset and turned them into a video by using their frame rates.

## Models for Detection

- For detection of Pearbags , 4 different models are used.
- Yolo8
- Yolo11
- Yolo8-CBAM-BiFPN
- Yolo8-CBAM

## Web App Details
### Features
Selected Model from Dropdown Menu\
Selected pearbags video for Detection and tracking

- Output : Detected and Counted Pearbags video and Evaluation metrics of selected model

## Running App

- Clone the repository and open on visual studio code and python run.py
