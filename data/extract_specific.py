import argparse
import os
import pdb
import shutil


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="coco/labels", help="path to parent dir with labels for splits")
    ap.add_argument("--class", default="person", dest="class_", help="class to extract labels for")
    ap.add_argument("--names", default="coco.names", help="file with the classes' names for the original dataset")
    args = ap.parse_args()

    # Get index for class from names file
    cls_idx = str(open(args.names).read().splitlines().index(args.class_))

    for split in ["train", "val"]:
        # Dirs with labels
        labels_dir = os.path.join(args.input, f"{split}2017")
        saved_labels_dir = os.path.join(args.input, f"{args.class_}_{split}2017")
        os.makedirs(saved_labels_dir, exist_ok=True)

        # Dirs with images
        images_dir = labels_dir.replace("labels", "images")
        saved_images_dir = saved_labels_dir.replace("labels", "images")
        os.makedirs(saved_images_dir, exist_ok=True)

        # pdb.set_trace()
        # Check each label file for the class and save the new labels
        for file in os.listdir(labels_dir):
            saved_lines = []
            path = os.path.join(labels_dir, file)

            with open(path, "r") as labels:
                for line in labels.read().splitlines():
                    lbl_cls = line.split(" ")[0]
                    if lbl_cls == cls_idx:
                        saved_lines.append(line)

            if len(saved_lines):
                new_path = os.path.join(saved_labels_dir, file)
                with open(new_path, "w") as new_lbl:
                    for saved_line in saved_lines:
                        new_lbl.write(f"{saved_line}\n")

        # Copy the images that contain the extracted labels to the new dir
        for file in os.listdir(saved_labels_dir):
            img = file.replace(".txt", ".jpg")
            shutil.copy(os.path.join(images_dir, img), saved_images_dir)