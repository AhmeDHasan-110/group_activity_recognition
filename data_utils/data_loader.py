import os
import glob
import pickle
from dataclasses import dataclass
from typing import Any

from generic_utils.config import DATA_CONFIG_PATH, SAVED_PICKLES_DIR, initialize_data_config

@dataclass
class PlayerBox:
    """Represents a single player's bounding box with its label and clip directory name."""
    label: str
    box: list[int]
    clip_dir: str

def load_annot(root: str, vid_dirs: list[str]) -> dict[str, dict[str, Any]]:
    """
    Loads bounding-box annotations and group labels from txt files
    
    Args:
        root (str): The path of the raw data(the videos directory).
        vid_dirs (list): The video directories that belong to a specific data subset. 
    """

    labels = {}

    for vid_dir in vid_dirs:
        cur_vid = os.path.join(root, str(vid_dir)) 

        txt_files = glob.glob(f'{cur_vid}/*.txt')
        if not txt_files:
            continue

        txt = txt_files[0]

        with open(txt, 'r') as file:
            for line in file:
                img_name, img_label, *boxes = line.split()

                key_name = os.path.join(str(vid_dir), os.path.splitext(img_name)[0])
                bboxes = [
                    PlayerBox(boxes[i+4].capitalize(), list(map(int, boxes[i : i+4])), key_name)
                    for i in range(0, len(boxes), 5)
                ]

                labels[key_name] = {
                    'group_label' : img_label.replace('-', '_'),
                    'player_labels' : bboxes
                }
    
    return labels

def save_to_pkl(data: Any, out_path: str, name: str) -> None:
    """Serializes and saves the data object to the specified directory as a pickle file"""

    os.makedirs(out_path, exist_ok=True)
    out_file = os.path.join(out_path, f"{name}.pkl")

    with open(out_file, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':

    cfg = initialize_data_config()

    root = cfg.root
    subsets = {
        'train_set' : cfg.train_subset,
        'validation_set' : cfg.val_subset,
        'test_set' : cfg.test_subset
        }

    for subset_name, vid_dirs in subsets.items():
        subset_out = os.path.join(SAVED_PICKLES_DIR, subset_name)
        os.makedirs(subset_out, exist_ok=True)

        labels = load_annot(root, vid_dirs)
        save_to_pkl(labels, subset_out, "labels")

        print(f"Saved: {subset_name}/labels.pkl")