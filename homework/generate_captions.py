import json
from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info

def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    # 1. Ego car
    # {kart_name} is the ego car.

    # 2. Counting
    # There are {num_karts} karts in the scenario.

    # 3. Track name
    # The track is {track_name}.

    # 4. Relative position
    # {kart_name} is {position} of the ego car.

 # Setup paths
    path_obj = Path(info_path)
    base_name = path_obj.stem.replace("_info", "")
    image_filename = f"{base_name}_{view_index:02d}_im.jpg"
    parent_dir = path_obj.parent.name
    image_rel_path = f"{parent_dir}/{image_filename}"

    # Extract data using Part 1 helpers
    track_name = extract_track_info(info_path)
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    ego_kart = next((k for k in karts if k["is_center_kart"]), None)

    captions = []

    # 1. Ego Identity
    if ego_kart:
        captions.append({
            "caption": f"{ego_kart['kart_name']} is the ego car.",
            "image_file": image_rel_path
        })

    # 2. Total Karts
    count = len(karts)
    k_str = "kart" if count == 1 else "karts"
    captions.append({
        "caption": f"There are {count} {k_str} in the scenario.",
        "image_file": image_rel_path
    })

    # 3. Track Name
    captions.append({
        "caption": f"The track is {track_name}.",
        "image_file": image_rel_path
    })

    if not ego_kart:
        return captions

    # 4. Relative Positions
    ego_x, ego_y = ego_kart["center"]

    for k in karts:
        if k["instance_id"] == ego_kart["instance_id"]:
            continue
            
        kx, ky = k["center"]
        k_name = k["kart_name"]
        
        # Horizontal (Left/Right)
        h_pos = "left" if kx < ego_x else "right"
        
        v_pos = "in front of" if ky < ego_y else "behind"
        

        captions.append({
            "caption": f"{k_name} is to the {h_pos} of the ego car.",
            "image_file": image_rel_path
        })
        captions.append({
            "caption": f"{k_name} is {v_pos} the ego car.",
            "image_file": image_rel_path
        })

    return captions

def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""

def generate_data(data_dir: str = "../data/train", output_file: str = "../data/train/train_captions.json"):
    """
    Bulk generate Captions for all files in a directory.
    """
    path = Path(data_dir)
    if not path.exists():
        print(f"Error: Data directory {path} does not exist.")
        return

    all_captions = []
    info_files = sorted(list(path.glob("*_info.json")))
    print(f"Found {len(info_files)} info files in {path}")
    
    for info_file in info_files:
        try:
            with open(info_file) as f:
                data = json.load(f)
                num_views = len(data.get("detections", []))
            
            for i in range(num_views):
                base_name = info_file.stem.replace("_info", "")
                img_name = f"{base_name}_{i:02d}_im.jpg"
                if (path / img_name).exists():
                    caps = generate_caption(str(info_file), i)
                    all_captions.extend(caps)
                    
        except Exception as e:
            print(f"Skipping {info_file}: {e}")
            continue
            
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_captions, f, indent=2)
        
    print(f"Generated {len(all_captions)} captions. Saved to {output_path}")


def main():
    fire.Fire({
        "check": check_caption,
        "generate": generate_data})     


if __name__ == "__main__":
    main()