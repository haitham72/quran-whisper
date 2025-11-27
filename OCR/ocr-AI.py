# python -m http.server 8000

import easyocr
import json
import glob
import os
from collections import defaultdict
from PIL import Image
import gc

# ---------------- CONFIG ----------------
pages_folder = "./pages"
output_folder = "./json"
os.makedirs(output_folder, exist_ok=True)

# Height normalization settings
TARGET_HEIGHT = 1500  # Target height for all pages (adjust as needed)

# Batch processing settings
BATCH_SIZE = 10  # Process N images at once (reduce if VRAM issues)
USE_GPU = True  # Set to False for CPU-only

# Initialize reader
print(f"Initializing EasyOCR (GPU: {USE_GPU})...")
reader = easyocr.Reader(['ar'], gpu=USE_GPU)

if USE_GPU:
    import torch
    if torch.cuda.is_available():
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"✓ VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  GPU requested but not available, using CPU")
        USE_GPU = False

# ---------------- HELPER FUNCTIONS ----------------
def get_bbox_info(box):
    """Extract x, y, width, height from 4-point box"""
    points = [[int(p[0]), int(p[1])] for p in box]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    return {
        'points': points,
        'x': min(xs),
        'y': min(ys),
        'width': max(xs) - min(xs),
        'height': max(ys) - min(ys),
        'center_y': sum(ys) / len(ys)
    }

def is_aya_separator(text):
    """Detect if text is an aya number separator (ornamental markers)"""
    # Common aya separators: ۝ ۞ ﴾ ﴿ or just numbers in ornamental form
    separators = ['۝', '۞', '﴾', '﴿', '﷽']
    
    # Check if it's a separator symbol
    if any(sep in text for sep in separators):
        return True
    
    # Check if it's just a number (aya number) - typically short and numeric
    text_clean = text.strip()
    if text_clean.isdigit() and len(text_clean) <= 3:
        return True
    
    # Check if it's very short (likely ornamental)
    if len(text_clean) <= 2 and not text_clean.isalpha():
        return True
    
    return False

def group_words_into_ayas(words, line_threshold=30):
    """Group words into ayas based on Y-coordinate (lines)"""
    if not words:
        return []
    
    # Sort words by Y position (top to bottom)
    sorted_words = sorted(words, key=lambda w: w['center_y'])
    
    ayas = []
    current_aya = [sorted_words[0]]
    
    for i in range(1, len(sorted_words)):
        prev_word = sorted_words[i-1]
        curr_word = sorted_words[i]
        
        # If Y difference is small, same line (same aya)
        y_diff = abs(curr_word['center_y'] - prev_word['center_y'])
        
        if y_diff < line_threshold:
            current_aya.append(curr_word)
        else:
            # New line = new aya
            ayas.append(current_aya)
            current_aya = [curr_word]
    
    # Add last aya
    if current_aya:
        ayas.append(current_aya)
    
    return ayas

def normalize_aya_heights(aya_words):
    """Normalize heights within an aya to be consistent"""
    if not aya_words:
        return aya_words
    
    # Calculate average height
    avg_height = sum(w['height'] for w in aya_words) / len(aya_words)
    
    # Adjust all words to average height
    for word in aya_words:
        mid_y = word['y'] + word['height'] / 2
        word['y'] = int(mid_y - avg_height / 2)
        word['height'] = int(avg_height)
        
        # Recalculate points based on new bbox
        x, y, w, h = word['x'], word['y'], word['width'], word['height']
        word['bbox_points'] = [
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ]
    
    return aya_words

# ---------------- PROCESS PAGES ----------------
page_files = sorted(glob.glob(os.path.join(pages_folder, "*.png")))

print(f"\nFound {len(page_files)} PNG files in '{pages_folder}/'")
print(f"Processing in batches of {BATCH_SIZE}\n")

if len(page_files) == 0:
    print("❌ ERROR: No PNG files found!")
    print(f"   Make sure images are in '{pages_folder}/' folder")
    exit(1)

# Process in batches
total_pages = len(page_files)
for batch_idx in range(0, total_pages, BATCH_SIZE):
    batch_files = page_files[batch_idx:batch_idx + BATCH_SIZE]
    batch_num = (batch_idx // BATCH_SIZE) + 1
    total_batches = (total_pages + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"{'='*60}")
    print(f"BATCH {batch_num}/{total_batches} (Pages {batch_idx+1}-{min(batch_idx+BATCH_SIZE, total_pages)})")
    print(f"{'='*60}\n")
    
    for img_path in batch_files:
        page_name = os.path.basename(img_path)
        print(f"Processing: {page_name}")
        
        # Load image to get dimensions
        img = Image.open(img_path)
        orig_width, orig_height = img.size
        
        # Calculate scale factor to normalize height
        scale_factor = TARGET_HEIGHT / orig_height
        
        print(f"  Original size: {orig_width}x{orig_height}")
        print(f"  Scale factor: {scale_factor:.3f}")
        
        # Run OCR
        print(f"  Running OCR...")
        results = reader.readtext(img_path)
        print(f"  OCR found {len(results)} text regions")
        
        if len(results) == 0:
            print(f"  ⚠️  WARNING: No text detected in {page_name}")
            continue
        
        # Extract word data with bbox info (normalized to target height)
        words = []
        separators = []  # Track separators separately
        
        for box, text, conf in results:
            bbox_info = get_bbox_info(box)
            
            word_data = {
                "text": text,
                "bbox_points": [[int(p[0] * scale_factor), int(p[1] * scale_factor)] 
                               for p in bbox_info['points']],
                "x": int(bbox_info['x'] * scale_factor),
                "y": int(bbox_info['y'] * scale_factor),
                "width": int(bbox_info['width'] * scale_factor),
                "height": int(bbox_info['height'] * scale_factor),
                "center_y": bbox_info['center_y'] * scale_factor,
                "confidence": float(conf)
            }
            
            # Separate aya markers from actual words
            if is_aya_separator(text):
                separators.append(word_data)
            else:
                words.append(word_data)
        
        # Group into ayas
        ayas = group_words_into_ayas(words)
        
        # Normalize heights within each aya
        normalized_ayas = []
        for aya_words in ayas:
            normalized_words = normalize_aya_heights(aya_words)
            normalized_ayas.append(normalized_words)
        
        # Flatten back to words list for JSON
        all_words = []
        for aya in normalized_ayas:
            all_words.extend(aya)
        
        # Prepare output
        output_data = {
            "metadata": {
                "original_width": orig_width,
                "original_height": orig_height,
                "normalized_height": TARGET_HEIGHT,
                "scale_factor": scale_factor
            },
            "words": [
                {
                    "text": w["text"],
                    "bbox_points": w["bbox_points"],
                    "confidence": w["confidence"]
                }
                for w in all_words
            ],
            "separators": [
                {
                    "text": s["text"],
                    "bbox_points": s["bbox_points"],
                    "confidence": s["confidence"]
                }
                for s in separators
            ],
            "ayas": [
                {
                    "words": [
                        {
                            "text": w["text"],
                            "bbox_points": w["bbox_points"],
                            "confidence": w["confidence"]
                        }
                        for w in aya
                    ],
                    "text": " ".join(w["text"] for w in aya)
                }
                for aya in normalized_ayas
            ]
        }
        
        # Save JSON
        json_path = os.path.join(output_folder, page_name.replace(".png", ".json"))
        with open(json_path, "w", encoding="utf8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ Saved: {json_path}")
        print(f"  ✓ Words: {len(words)}, Ayas: {len(normalized_ayas)}, Separators: {len(separators)}\n")
        
        # Clean up image
        img.close()
    
    # Clear GPU cache after each batch
    if USE_GPU:
        try:
            import torch
            torch.cuda.empty_cache()
            print(f"✓ Cleared GPU cache after batch {batch_num}\n")
        except:
            pass
    
    # Force garbage collection
    gc.collect()

print(f"\n{'='*60}")
print("✅ All pages processed!")
print(f"📁 JSON files saved in: {output_folder}/")
print(f"   Total files: {len(glob.glob(os.path.join(output_folder, '*.json')))}")
print(f"{'='*60}")