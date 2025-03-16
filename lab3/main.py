from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from array import array
import os
from PIL import Image
import sys
import time
import pytesseract
import Levenshtein
from difflib import SequenceMatcher
import numpy as np
import cv2
import io

from lib.keys import VISION_KEY, VISION_ENDPOINT
from utils.functions import evaluate_ocr_quality, evaluate_text_localization, evaluate_spatial_alignment, process_image_for_ocr

def evaluate_image(image, ground_truth_text, ground_truth_boxes, name=""):
    """
    Evaluate an image using both Azure and Tesseract
    """
    print(f"\n{'='*20} Evaluare {name} {'='*20}")
    
    # Convert image for Azure API
    is_success, buffer = cv2.imencode(".jpg", image)
    io_buf = io.BytesIO(buffer.tobytes())
    
    # Azure OCR
    read_response = computervision_client.read_in_stream(
        image=io_buf,
        mode="Printed",
        raw=True
    )
    
    operation_id = read_response.headers['Operation-Location'].split('/')[-1]
    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)
    
    # Process Azure results
    azure_results = []
    azure_boxes = []
    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results:
            for line in text_result.lines:
                azure_results.append(line.text)
                bbox = line.bounding_box
                azure_boxes.append([
                    bbox[0], bbox[1],
                    bbox[4], bbox[5]
                ])
    
    azure_text = " ".join(azure_results)
    print(f"\nText detectat de Azure:")
    print(azure_text)
    
    # Tesseract OCR
    tesseract_data = pytesseract.image_to_data(image, lang="ron", output_type=pytesseract.Output.DICT)
    tesseract_boxes = []
    tesseract_lines = []
    
    for i in range(len(tesseract_data['text'])):
        if int(tesseract_data['conf'][i]) > 0:
            x = tesseract_data['left'][i]
            y = tesseract_data['top'][i]
            w = tesseract_data['width'][i]
            h = tesseract_data['height'][i]
            tesseract_boxes.append([x, y, x + w, y + h])
            tesseract_lines.append(tesseract_data['text'][i])
    
    tesseract_text = " ".join(tesseract_lines)
    print(f"\nText detectat de Tesseract:")
    print(tesseract_text)
    
    # Evaluate results
    print("\n--- Evaluare Azure ---")
    azure_metrics = evaluate_ocr_quality([azure_text], [ground_truth_text])
    azure_loc_metrics = evaluate_text_localization(azure_boxes, ground_truth_boxes)
    azure_spatial_metrics = evaluate_spatial_alignment(azure_boxes, ground_truth_boxes)
    
    print("\nCalitatea textului:")
    print("La nivel de caracter:")
    for metric, value in azure_metrics['char_level'][-1].items():
        print(f"{metric}: {value:.4f}")
    print("\nLa nivel de cuvânt:")
    for metric, value in azure_metrics['word_level'][-1].items():
        print(f"{metric}: {value:.4f}")
    
    print("\nCalitatea localizării:")
    print(f"Mean IoU: {azure_loc_metrics['mean_iou']:.4f}")
    print(f"Detection Rate: {azure_loc_metrics['detection_rate']:.4f}")
    print(f"False Positive Rate: {azure_loc_metrics['false_positive_rate']:.4f}")
    print(f"Vertical Alignment: {azure_spatial_metrics['vertical_alignment_score']:.4f}")
    print(f"Horizontal Alignment: {azure_spatial_metrics['horizontal_alignment_score']:.4f}")
    
    print("\n--- Evaluare Tesseract ---")
    tesseract_metrics = evaluate_ocr_quality([tesseract_text], [ground_truth_text])
    tesseract_loc_metrics = evaluate_text_localization(tesseract_boxes, ground_truth_boxes)
    tesseract_spatial_metrics = evaluate_spatial_alignment(tesseract_boxes, ground_truth_boxes)
    
    print("\nCalitatea textului:")
    print("La nivel de caracter:")
    for metric, value in tesseract_metrics['char_level'][-1].items():
        print(f"{metric}: {value:.4f}")
    print("\nLa nivel de cuvânt:")
    for metric, value in tesseract_metrics['word_level'][-1].items():
        print(f"{metric}: {value:.4f}")
    
    print("\nCalitatea localizării:")
    print(f"Mean IoU: {tesseract_loc_metrics['mean_iou']:.4f}")
    print(f"Detection Rate: {tesseract_loc_metrics['detection_rate']:.4f}")
    print(f"False Positive Rate: {tesseract_loc_metrics['false_positive_rate']:.4f}")
    print(f"Vertical Alignment: {tesseract_spatial_metrics['vertical_alignment_score']:.4f}")
    print(f"Horizontal Alignment: {tesseract_spatial_metrics['horizontal_alignment_score']:.4f}")
    
    return {
        'azure': {
            'text': azure_text,
            'metrics': azure_metrics,
            'loc_metrics': azure_loc_metrics,
            'spatial_metrics': azure_spatial_metrics
        },
        'tesseract': {
            'text': tesseract_text,
            'metrics': tesseract_metrics,
            'loc_metrics': tesseract_loc_metrics,
            'spatial_metrics': tesseract_spatial_metrics
        }
    }

# -------------------------- MAIN --------------------------
'''
Authenticate
Authenticates your credentials and creates a client.
'''
subscription_key = VISION_KEY
endpoint = VISION_ENDPOINT
print(subscription_key)
print(endpoint)

computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
'''
END - Authenticate
'''

# Load and process image
image_path = "images/test2.jpeg"
original_image = cv2.imread(image_path)
processed_image = process_image_for_ocr(image_path)

# Display images
cv2.imshow('Original Image', original_image)
cv2.imshow('Processed Image', processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Ground truth data
groundTruth = ["Succes in rezolvarea", "tEMELOR la", "LABORAtoaree de", "Inteligenta Artificiala!"]
ground_truth_text = " ".join(groundTruth)
groundTruth_boxes = [
    [100, 50, 400, 100],
    [150, 120, 350, 170],
    [120, 190, 380, 240],
    [130, 260, 370, 310]
]

# Evaluate both images
original_results = evaluate_image(original_image, ground_truth_text, groundTruth_boxes, "IMAGINE ORIGINALĂ")
processed_results = evaluate_image(processed_image, ground_truth_text, groundTruth_boxes, "IMAGINE PROCESATĂ")

# Print comparison between original and processed
print("\n" + "="*20 + " COMPARAȚIE FINALĂ " + "="*20)

def print_comparison(metric_name, original_value, processed_value):
    diff = processed_value - original_value
    arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
    print(f"{metric_name}:")
    print(f"  Original:  {original_value:.4f}")
    print(f"  Procesat:  {processed_value:.4f}")
    print(f"  Diferență: {diff:.4f} {arrow}")

print("\nAzure Computer Vision:")
for metric in original_results['azure']['metrics']['char_level'][-1].keys():
    print_comparison(
        metric,
        original_results['azure']['metrics']['char_level'][-1][metric],
        processed_results['azure']['metrics']['char_level'][-1][metric]
    )

print("\nTesseract:")
for metric in original_results['tesseract']['metrics']['char_level'][-1].keys():
    print_comparison(
        metric,
        original_results['tesseract']['metrics']['char_level'][-1][metric],
        processed_results['tesseract']['metrics']['char_level'][-1][metric]
    )

print("\nLocalizare text:")
metrics_loc = [
    ('Mean IoU', 'mean_iou'),
    ('Detection Rate', 'detection_rate'),
    ('False Positive Rate', 'false_positive_rate'),
    ('Vertical Alignment', 'vertical_alignment_score'),
    ('Horizontal Alignment', 'horizontal_alignment_score')
]

print("\nAzure:")
for name, key in metrics_loc:
    if key in original_results['azure']['loc_metrics']:
        print_comparison(
            name,
            original_results['azure']['loc_metrics'][key],
            processed_results['azure']['loc_metrics'][key]
        )
    else:
        print_comparison(
            name,
            original_results['azure']['spatial_metrics'][key],
            processed_results['azure']['spatial_metrics'][key]
        )

print("\nTesseract:")
for name, key in metrics_loc:
    if key in original_results['tesseract']['loc_metrics']:
        print_comparison(
            name,
            original_results['tesseract']['loc_metrics'][key],
            processed_results['tesseract']['loc_metrics'][key]
        )
    else:
        print_comparison(
            name,
            original_results['tesseract']['spatial_metrics'][key],
            processed_results['tesseract']['spatial_metrics'][key]
        )