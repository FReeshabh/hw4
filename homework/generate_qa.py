import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """

    with open(info_path, 'r') as f:
        data = json.load(f)

    if view_index >= len(data["detections"]):
        return []

    detections = data["detections"][view_index]
    karts_metadata = data.get("karts", [])
    
    kart_objects = []
    # Scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    for det in detections:
        class_id, track_id, x1, y1, x2, y2 = det
        class_id = int(class_id)
        track_id = int(track_id)

        # We only care about Karts (class_id 1)
        if class_id != 1:
            continue

        # Scale to current image size
        sx1, sy1 = x1 * scale_x, y1 * scale_y
        sx2, sy2 = x2 * scale_x, y2 * scale_y
        
        # Filter small/invalid boxes
        if (sx2 - sx1) < min_box_size or (sy2 - sy1) < min_box_size:
            continue

        # Calculate center
        cx = (sx1 + sx2) / 2
        cy = (sy1 + sy2) / 2

        # Get Kart Name
        kart_name = "unknown"
        if karts_metadata and track_id < len(karts_metadata):
            item = karts_metadata[track_id]
            if isinstance(item, dict):
                kart_name = item.get("name", "unknown")
            elif isinstance(item, str):
                kart_name = item

        # Identify Ego Car (Track ID 0)
        is_ego = (track_id == 0)

        kart_objects.append({
            "instance_id": track_id,
            "kart_name": kart_name,
            "center": (cx, cy),
            "is_center_kart": is_ego
        })

    return kart_objects

def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """

    # === FIX: Actually read the JSON file ===
    with open(info_path, 'r') as f:
        data = json.load(f)
    return data.get('track', data.get('track_name', 'unknown'))

def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    # 1. Ego car question
    # What kart is the ego car?

    # 2. Total karts question
    # How many karts are there in the scenario?

    # 3. Track information questions
    # What track is this?

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or back the ego car?
    # Where is {kart_name} relative to the ego car?

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are back the ego car?

     # get the paths right
    path_obj = Path(info_path)
    base_name = path_obj.stem.replace("_info", "")
    image_filename = f"{base_name}_{view_index:02d}_im.jpg"
    
    # The 'image_file' field in JSON usually wants relative path like "train/00000_00_im.jpg"
    parent_dir = path_obj.parent.name # e.g., "train"
    image_rel_path = f"{parent_dir}/{image_filename}"
    
    track_name = extract_track_info(info_path)
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)

    # Find Ego Kart
    ego_kart = next((k for k in karts if k["is_center_kart"]), None)
    
    qa_pairs = []

    # Q1: Ego car Identity
    if ego_kart:
        qa_pairs.append({
            "question": "What kart is the ego car?",
            "answer": str(ego_kart["kart_name"]),
            "image_file": image_rel_path
        })

    # Q2: Total Karts
    qa_pairs.append({
        "question": "How many karts are there in the scenario?",
        "answer": str(len(karts)),
        "image_file": image_rel_path
    })

    # Q3: Track Name
    qa_pairs.append({
        "question": "What track is this?",
        "answer": str(track_name),
        "image_file": image_rel_path
    })

    if not ego_kart:
        return qa_pairs

    # Relative Position Logic
    ego_x, ego_y = ego_kart["center"]
    
    # Counters
    left_count = 0
    right_count = 0
    front_count = 0
    back_count = 0


    for k in karts:
        if k["instance_id"] == ego_kart["instance_id"]:
            continue
            
        kx, ky = k["center"]
        k_name = k["kart_name"]
        
        h_pos = "left" if kx < ego_x else "right"
        if h_pos == "left": left_count += 1
        else: right_count += 1
        
        v_pos = "front" if ky < ego_y else "back"
        if v_pos == "front": front_count += 1
        else: back_count += 1

        qa_pairs.append({
            "question": f"Is {k_name} to the left or right of the ego car?",
            "answer": h_pos,
            "image_file": image_rel_path
        })
        qa_pairs.append({
            "question": f"Is {k_name} in front of or back the ego car?",
            "answer": v_pos,  # Returns "front" or "back"
            "image_file": image_rel_path
        })
        qa_pairs.append({
            "question": f"Where is {k_name} relative to the ego car?",
            # Standard Format: Space separator (e.g., "front right")
            "answer": f"{v_pos} {h_pos}", 
            "image_file": image_rel_path
        })

    qa_pairs.append({
        "question": "How many karts are to the left of the ego car?",
        "answer": str(left_count),
        "image_file": image_rel_path
    })
    qa_pairs.append({
        "question": "How many karts are to the right of the ego car?",
        "answer": str(right_count),
        "image_file": image_rel_path
    })
    qa_pairs.append({
        "question": "How many karts are in front of the ego car?",
        "answer": str(front_count),
        "image_file": image_rel_path
    })
    qa_pairs.append({
        "question": "How many karts are back the ego car?",
        "answer": str(back_count),
        "image_file": image_rel_path
    })

    return qa_pairs
def generate_data(data_dir: str = "data/train", output_file: str = "data/train/train_qa_pairs.json"):
    """
    Bulk generate QA pairs for all files in a directory.
    """
    path = Path(data_dir)
    all_qa_pairs = []
    
    # Get all info files
    info_files = sorted(list(path.glob("*_info.json")))
    print(f"Found {len(info_files)} info files in {data_dir}")
    
    for info_file in info_files:
        try:
            # Read how many views this file has
            with open(info_file) as f:
                data = json.load(f)
                num_views = len(data.get("detections", []))
            
            # Generate pairs for every view in this file
            for i in range(num_views):
                # Check if the image actually exists before generating
                base_name = info_file.stem.replace("_info", "")
                img_name = f"{base_name}_{i:02d}_im.jpg"
                if (path / img_name).exists():
                    # Call your existing single-image function
                    pairs = generate_qa_pairs(str(info_file), i)
                    all_qa_pairs.extend(pairs)
                    
        except Exception as e:
            print(f"Error processing {info_file}: {e}")
            continue
            
    # Save the huge list to a JSON file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_qa_pairs, f, indent=2)
        
    print(f"Successfully generated {len(all_qa_pairs)} QA pairs saved to {output_file}")

def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_qa_pairs,
    "generate": generate_data})


if __name__ == "__main__":
    main()