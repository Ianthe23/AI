from lib.keys import VISION_KEY, VISION_ENDPOINT
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes, VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import matplotlib.pyplot as plt
import os
import time
from matplotlib.widgets import RectangleSelector
import numpy as np

subscription_key = VISION_KEY
endpoint = VISION_ENDPOINT
print(subscription_key)
print(endpoint)

computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

# Global variables for manual annotations
manual_boxes = []

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    box format: [x1, y1, x2, y2]
    """
    # Get coordinates of intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate area of intersection
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union area
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def onselect(eclick, erelease):
    'eclick and erelease are the press and release events'
    global start_x, start_y, end_x, end_y
    start_x, start_y = eclick.xdata, eclick.ydata
    end_x, end_y = erelease.xdata, erelease.ydata
    box = [min(start_x, end_x), min(start_y, end_y), max(start_x, end_x), max(start_y, end_y)]
    manual_boxes.append(box)
    print(f'Selected coordinates: ({start_x:.1f}, {start_y:.1f}) to ({end_x:.1f}, {end_y:.1f})')

def manual_annotation(image_path):
    """
    Manual annotation of bicycles in an image
    Returns the time taken for annotation
    """
    global manual_boxes
    manual_boxes = []  # Reset manual boxes list
    
    img_array = plt.imread(image_path)
    fig, ax = plt.subplots()
    ax.imshow(img_array)
    
    # Initialize rectangle selector with correct parameters
    rs = RectangleSelector(ax, onselect, 
                          button=[1],  # Left mouse button
                          minspanx=5, minspany=5,  # Minimum size of selection
                          spancoords='pixels')  # Use pixels for coordinates
    
    plt.title('Click and drag to draw boxes around bicycles. Press Enter when done.')
    plt.show()
    
    return rs

def classify_image(image_path):
    """
    Classify an image as containing a bicycle or not
    Returns the confidence score
    """
    img = open(image_path, "rb")
    result = computervision_client.analyze_image_in_stream(img, visual_features=[VisualFeatureTypes.tags])
    
    # Check for bicycle-related tags
    bicycle_tags = ["bicycle", "bike", "cycling"]
    max_confidence = 0
    
    for tag in result.tags:
        if tag.name.lower() in bicycle_tags:
            max_confidence = max(max_confidence, tag.confidence)
    
    # Return prediction (1 for bicycle, 0 for no bicycle) and confidence
    return 1 if max_confidence > 0.5 else 0, max_confidence

def detect_bicycles(image_path):
    """
    Automatic bicycle detection using Azure Computer Vision
    Returns the time taken for detection
    """
    start_time = time.time()
    
    img = open(image_path, "rb")
    result = computervision_client.analyze_image_in_stream(img, visual_features=[VisualFeatureTypes.objects])
    
    # Get image dimensions
    img_array = plt.imread(image_path)
    height, width = img_array.shape[:2]
    
    # Create figure and display image
    plt.figure(figsize=(10, 8))
    plt.imshow(img_array)
    
    # Colors for different bicycles
    colors = ['red', 'blue', 'green', 'yellow', 'purple']
    bike_count = 0
    
    # Draw bounding boxes for bicycles
    for obj in result.objects:
        if obj.object_property.lower() in ["bicycle", "bike", "bicycle frame", "cycle"]:
            # Get bounding box coordinates
            x = obj.rectangle.x
            y = obj.rectangle.y
            w = obj.rectangle.w
            h = obj.rectangle.h
            
            # Use different colors for different bicycles
            color = colors[bike_count % len(colors)]
            
            # Draw rectangle
            rect = plt.Rectangle((x, y), w, h, fill=False, color=color, linewidth=2)
            plt.gca().add_patch(rect)
            
            # Add label with bicycle number
            plt.text(x, y-10, f'Bicycle {bike_count + 1} ({obj.confidence:.2f})', 
                    color=color, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
            
            bike_count += 1
    
    plt.title(f'Automatic Bicycle Detection - Found {bike_count} bicycles')
    plt.axis('off')
    plt.show()
    
    end_time = time.time()
    return end_time - start_time

def compare_annotation_methods(image_path):
    """
    Compare automatic and manual annotation methods
    """
    print("\n=== Automatic Detection ===")
    auto_time = detect_bicycles(image_path)
    print(f"Automatic detection took: {auto_time:.2f} seconds")
    
    print("\n=== Manual Annotation ===")
    print("Please annotate the bicycles manually. Click and drag to draw boxes, press Enter when done.")
    manual_time = time.time()
    rs = manual_annotation(image_path)
    manual_time = time.time() - manual_time
    print(f"Manual annotation took: {manual_time:.2f} seconds")
    
    print(f"\nTime difference: {abs(auto_time - manual_time):.2f} seconds")
    print(f"Automatic detection was {'faster' if auto_time < manual_time else 'slower'} than manual annotation")

def get_automatic_boxes(image_path):
    """
    Get automatic detection boxes from Azure Computer Vision
    """
    img = open(image_path, "rb")
    result = computervision_client.analyze_image_in_stream(img, visual_features=[VisualFeatureTypes.objects])
    
    auto_boxes = []
    for obj in result.objects:
        if obj.object_property.lower() in ["bicycle", "bike", "bicycle frame", "cycle"]:
            x = obj.rectangle.x
            y = obj.rectangle.y
            w = obj.rectangle.w
            h = obj.rectangle.h
            box = [x, y, x + w, y + h]
            auto_boxes.append(box)
    
    return auto_boxes

def evaluate_detection_performance(image_path):
    """
    Evaluate automatic detection performance against manual annotations
    """
    # Get automatic detections
    auto_boxes = get_automatic_boxes(image_path)
    
    # Get manual annotations
    print("\nPlease annotate the bicycles manually:")
    manual_annotation(image_path)
    
    # Calculate metrics
    total_boxes = len(manual_boxes)
    detected_boxes = len(auto_boxes)
    
    # Calculate IoU for each manual box with its best matching automatic box
    iou_scores = []
    matched_auto_boxes = set()
    
    for manual_box in manual_boxes:
        best_iou = 0
        best_match_idx = -1
        
        for i, auto_box in enumerate(auto_boxes):
            if i not in matched_auto_boxes:
                iou = calculate_iou(manual_box, auto_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = i
        
        if best_match_idx != -1:
            matched_auto_boxes.add(best_match_idx)
            iou_scores.append(best_iou)
    
    # Calculate metrics
    mean_iou = np.mean(iou_scores) if iou_scores else 0
    detection_rate = len(matched_auto_boxes) / total_boxes if total_boxes > 0 else 0
    false_positives = detected_boxes - len(matched_auto_boxes)
    
    # Display results
    print("\nDetection Performance Metrics:")
    print(f"Total manual annotations: {total_boxes}")
    print(f"Total automatic detections: {detected_boxes}")
    print(f"Mean IoU: {mean_iou:.3f}")
    print(f"Detection Rate: {detection_rate:.3f}")
    print(f"False Positives: {false_positives}")
    
    # Visualize results
    img_array = plt.imread(image_path)
    plt.figure(figsize=(12, 8))
    plt.imshow(img_array)
    
    # Draw manual boxes in blue
    for box in manual_boxes:
        rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                           fill=False, color='blue', linewidth=2, label='Manual')
        plt.gca().add_patch(rect)
    
    # Draw automatic boxes in red
    for box in auto_boxes:
        rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                           fill=False, color='red', linewidth=2, label='Automatic')
        plt.gca().add_patch(rect)
    
    plt.title('Comparison of Manual (Blue) and Automatic (Red) Detections')
    plt.legend()
    plt.axis('off')
    plt.show()