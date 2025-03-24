from lib.keys import VISION_KEY, VISION_ENDPOINT
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes, VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from utils.functions import classify_image, detect_bicycles, compare_annotation_methods, evaluate_detection_performance

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

def binary_classification():
    # Test the classifier
    test_images_dir = "images"
    true_labels = []  # 1 for bicycle images, 0 for non-bicycle images
    predictions = []

    # Process all images in the directory
    for filename in os.listdir(test_images_dir):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(test_images_dir, filename)
            # Determine true label based on filename (assuming files with 'bike' in name are bicycle images)
            true_label = 1 if "bike" in filename.lower() else 0
            true_labels.append(true_label)
            
            # Get prediction
            pred, confidence = classify_image(image_path)
            predictions.append(pred)
            
            print(f"Image: {filename}")
            print(f"True label: {'Bicycle' if true_label == 1 else 'No bicycle'}")
            print(f"Prediction: {'Bicycle' if pred == 1 else 'No bicycle'}")
            print(f"Confidence: {confidence:.2f}")
            print("---")

    # Calculate performance metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    conf_matrix = confusion_matrix(true_labels, predictions)

    print("\nPerformance Metrics:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def object_detection():
    # Test the object detector
    test_images_dir = "images"
    for filename in os.listdir(test_images_dir):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(test_images_dir, filename)
            detect_bicycles(image_path)

def main():
    # binary_classification()
    # object_detection()
    # compare_annotation_methods("images/bike02.jpg")
    evaluate_detection_performance("images/bike05.jpg")


if __name__ == "__main__":
    main()

