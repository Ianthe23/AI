import Levenshtein
from difflib import SequenceMatcher
import numpy as np
import cv2
import os

# -------------------------- FUNCTIONS --------------------------
def hamming_distance(s1, s2):
    """Calculate the Hamming distance between two strings"""
    if len(s1) != len(s2):
        return len(s1) if len(s2) == 0 else len(s2) if len(s1) == 0 else max(len(s1), len(s2))
    
    
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def evaluate_text_similarity(predicted_text, ground_truth, level='word'):
    """
    Evaluate text similarity using multiple metrics
    level: 'word' or 'char' for word-level or character-level analysis
    """
    if level == 'word':
        pred_tokens = predicted_text.split()
        truth_tokens = ground_truth.split()
    else:
        pred_tokens = list(predicted_text)
        truth_tokens = list(ground_truth)
    
    # Calculate various distances
    levenshtein_dist = Levenshtein.distance(predicted_text, ground_truth) 
    jaro_winkler_sim = Levenshtein.jaro_winkler(predicted_text, ground_truth)
    hamming_dist = hamming_distance(predicted_text, ground_truth)
    
    # Longest Common Subsequence using SequenceMatcher
    lcs = SequenceMatcher(None, predicted_text, ground_truth).find_longest_match(
        0, len(predicted_text), 0, len(ground_truth))
    lcs_length = lcs.size
    
    # Normalize distances
    max_len = max(len(predicted_text), len(ground_truth))
    normalized_levenshtein = 1 - (levenshtein_dist / max_len if max_len > 0 else 0)
    normalized_hamming = 1 - (hamming_dist / max_len if max_len > 0 else 0)
    normalized_lcs = lcs_length / max_len if max_len > 0 else 0
    
    return {
        'levenshtein_similarity': normalized_levenshtein,
        'jaro_winkler_similarity': jaro_winkler_sim,
        'hamming_similarity': normalized_hamming,
        'lcs_similarity': normalized_lcs
    }

def evaluate_ocr_quality(predicted_texts, ground_truths):
    """
    Evaluate OCR quality for both character and word level
    """
    results = {
        'char_level': [],
        'word_level': []
    }
    
    for pred, truth in zip(predicted_texts, ground_truths):
        char_metrics = evaluate_text_similarity(pred, truth, level='char')
        word_metrics = evaluate_text_similarity(pred, truth, level='word')
        
        results['char_level'].append(char_metrics)
        results['word_level'].append(word_metrics)
    
    # Calculate averages
    for level in results:
        avg_metrics = {}
        for metric in results[level][0].keys():
            values = [doc[metric] for doc in results[level]]
            avg_metrics[f'avg_{metric}'] = np.mean(values)
        results[level].append(avg_metrics)
    
    return results

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union between two bounding boxes
    box format: [x1, y1, x2, y2] where (x1,y1) is top-left and (x2,y2) is bottom-right
    """
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    if x2 < x1 or y2 < y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0.0

def evaluate_text_localization(predicted_boxes, ground_truth_boxes, iou_threshold=0.5):
    """
    Evaluate text localization quality using IoU
    predicted_boxes and ground_truth_boxes: list of [x1, y1, x2, y2] coordinates
    """
    results = {
        'mean_iou': 0.0,
        'detection_rate': 0.0,
        'false_positive_rate': 0.0,
        'matches': []
    }
    
    if not ground_truth_boxes or not predicted_boxes:
        return results
    
    # Calculate IoU for each pair of boxes
    ious = np.zeros((len(predicted_boxes), len(ground_truth_boxes)))
    for i, pred_box in enumerate(predicted_boxes):
        for j, gt_box in enumerate(ground_truth_boxes):
            ious[i,j] = calculate_iou(pred_box, gt_box)
    
    # Find matches using IoU threshold
    matches = []
    used_preds = set()
    used_gts = set()
    
    # Find best matches
    while True:
        if np.max(ious) < iou_threshold:
            break
        pred_idx, gt_idx = np.unravel_index(np.argmax(ious), ious.shape)
        if pred_idx in used_preds or gt_idx in used_gts:
            ious[pred_idx, gt_idx] = 0
            continue
        
        matches.append({
            'pred_idx': pred_idx,
            'gt_idx': gt_idx,
            'iou': ious[pred_idx, gt_idx]
        })
        used_preds.add(pred_idx)
        used_gts.add(gt_idx)
        ious[pred_idx, gt_idx] = 0
    
    # Calculate metrics
    results['matches'] = matches
    results['mean_iou'] = np.mean([m['iou'] for m in matches]) if matches else 0.0
    results['detection_rate'] = len(matches) / len(ground_truth_boxes)
    results['false_positive_rate'] = (len(predicted_boxes) - len(matches)) / len(predicted_boxes) if predicted_boxes else 0.0
    
    return results

def evaluate_spatial_alignment(predicted_boxes, ground_truth_boxes):
    """
    Evaluate the spatial alignment of text (relative positioning)
    """
    if not predicted_boxes or not ground_truth_boxes:
        return {
            'vertical_alignment_score': 0.0,
            'horizontal_alignment_score': 0.0
        }
    
    def get_center(box):
        return [(box[0] + box[2])/2, (box[1] + box[3])/2]
    
    # Calculate centers
    pred_centers = [get_center(box) for box in predicted_boxes]
    gt_centers = [get_center(box) for box in ground_truth_boxes]
    
    # Calculate relative positions
    def calculate_relative_positions(centers):
        n = len(centers)
        positions = []
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[j][0] - centers[i][0]
                dy = centers[j][1] - centers[i][1]
                positions.append((dx, dy))
        return positions
    
    pred_positions = calculate_relative_positions(pred_centers)
    gt_positions = calculate_relative_positions(gt_centers)
    
    # Calculate alignment scores
    def calculate_alignment_score(pred_pos, gt_pos):
        if not pred_pos or not gt_pos:
            return 0.0
        
        # Normalize positions
        def normalize_positions(positions):
            if not positions:
                return []
            max_dist = max(max(abs(dx), abs(dy)) for dx, dy in positions)
            if max_dist == 0:
                return positions
            return [(dx/max_dist, dy/max_dist) for dx, dy in positions]
        
        pred_pos_norm = normalize_positions(pred_pos)
        gt_pos_norm = normalize_positions(gt_pos)
        
        # Calculate minimum differences
        min_diffs = []
        for pp in pred_pos_norm:
            diffs = [np.sqrt((pp[0]-gp[0])**2 + (pp[1]-gp[1])**2) for gp in gt_pos_norm]
            min_diffs.append(min(diffs))
        
        return 1 - min(1, np.mean(min_diffs))
    
    # Separate horizontal and vertical alignment
    pred_horizontal = [(dx, 0) for dx, dy in pred_positions]
    gt_horizontal = [(dx, 0) for dx, dy in gt_positions]
    pred_vertical = [(0, dy) for dx, dy in pred_positions]
    gt_vertical = [(0, dy) for dx, dy in gt_positions]
    
    return {
        'vertical_alignment_score': calculate_alignment_score(pred_vertical, gt_vertical),
        'horizontal_alignment_score': calculate_alignment_score(pred_horizontal, gt_horizontal)
    }

def process_image_for_ocr(image_path):
    """
    Process an image for OCR by applying a series of preprocessing steps
    image_path: path to the image file

    Returns:
        processed_image: processed image as a numpy array
    """
    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize image size (resize if too large or small)
    height, width = gray.shape
    max_dimension = 2000
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Normalize pixel values to [0, 255]
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # # Apply Gaussian blur to reduce noise before thresholding
    # blurred = cv2.GaussianBlur(gray, (3,3), 0)

    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # # Denoise using Non-Local Means Denoising
    # denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)

    # # Enhance contrast using CLAHE
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # enhanced = clahe.apply(gray)

    # # Deskew
    # coords = np.column_stack(np.where(enhanced > 0))
    # if len(coords) > 0:  # Check if there are any non-zero points
    #     angle = cv2.minAreaRect(coords)[-1]
    #     if angle < -45:
    #         angle = 90 + angle
    #     center = (enhanced.shape[1] // 2, enhanced.shape[0] // 2)
    #     M = cv2.getRotationMatrix2D(center, angle, 1.0)
    #     rotated = cv2.warpAffine(
    #         enhanced,
    #         M,
    #         (enhanced.shape[1], enhanced.shape[0]),
    #         flags=cv2.INTER_CUBIC,
    #         borderMode=cv2.BORDER_REPLICATE
    #     )
    # else:
    #     rotated = enhanced

    # Final normalization
    rotated = cv2.normalize(binary, None, 0, 255, cv2.NORM_MINMAX)

    # Convert back to BGR for display
    processed_image = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)

    # Save intermediate results for visualization
    output_dir = "images/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(f"{output_dir}/{base_name}_gray.jpg", gray)
    cv2.imwrite(f"{output_dir}/{base_name}_binary.jpg", binary)
    # cv2.imwrite(f"{output_dir}/{base_name}_denoised.jpg", denoised)
    # cv2.imwrite(f"{output_dir}/{base_name}_enhanced.jpg", enhanced)
    cv2.imwrite(f"{output_dir}/{base_name}_rotated.jpg", rotated)
    cv2.imwrite(f"{output_dir}/{base_name}_final.jpg", processed_image)

    return processed_image


    
