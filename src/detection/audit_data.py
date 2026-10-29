import cv2
import random
import matplotlib.pyplot as plt
import hashlib

from PIL import Image

from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "Data"
DETECTION_DIR = DATA_DIR / "detection"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
RANDOM_STATE = 42

EXPECTED_SPLITS = {"Train", "Val"}
EXPECTED_CLASSES = {"Glioma", "Meningioma", "No Tumor", "Pituitary"}


def get_detection_splits(detection_dir: Path):
    """Return the available dataset splits."""
    if not detection_dir.is_dir():
        print(f"Detection directory not found: {detection_dir}")
        return []

    return sorted(
        folder
        for folder in detection_dir.iterdir()
        if folder.is_dir() and folder.name in EXPECTED_SPLITS
    )


def get_detection_class_dirs(split_dir: Path):
    """Return the class directories inside a split."""
    return sorted(
        folder
        for folder in split_dir.iterdir()
        if folder.is_dir() and folder.name in EXPECTED_CLASSES
    )


def get_detection_duplicate_images(splits):
    """Return groups of detection images with identical file contents."""
    image_hashes = {}

    for split_dir in splits:
        for class_dir in get_detection_class_dirs(split_dir):
            images_dir = class_dir / "images"

            if not images_dir.is_dir():
                continue

            for image_path in sorted(images_dir.iterdir()):
                if (
                    not image_path.is_file()
                    or image_path.suffix.lower() not in VALID_EXTENSIONS
                ):
                    continue

                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                image_hashes.setdefault(digest, []).append(image_path)

    return {
        digest: paths for digest, paths in image_hashes.items() if len(paths) > 1
    }


def explore_detection_structure(detection_dir: Path):
    """Explore and validate the detection dataset structure."""
    print("Detection directory:", detection_dir)
    print("Detection directory exists:", detection_dir.is_dir())

    if not detection_dir.is_dir():
        return []

    actual_splits = {
        folder.name for folder in detection_dir.iterdir() if folder.is_dir()
    }

    print("\nNumber of splits:", len(actual_splits))
    print("Available splits:", sorted(actual_splits))

    print("Missing splits:", sorted(EXPECTED_SPLITS - actual_splits))
    print("Unexpected directories:", sorted(actual_splits - EXPECTED_SPLITS))

    splits = get_detection_splits(detection_dir)

    for split_dir in splits:
        class_dirs = get_detection_class_dirs(split_dir)
        actual_classes = {folder.name for folder in class_dirs}

        print(f"\nSplit: {split_dir.name}")
        print("Number of classes:", len(actual_classes))
        print("Available classes:", sorted(actual_classes))
        print("Missing classes:", sorted(EXPECTED_CLASSES - actual_classes))
        print("Unexpected classes:", sorted(actual_classes - EXPECTED_CLASSES))

        for class_dir in class_dirs:
            images_dir = class_dir / "images"
            labels_dir = class_dir / "labels"

            print(f"\n  Class: {class_dir.name}")
            print(f"    Images directory exists: {images_dir.is_dir()}")
            print(f"    Labels directory exists: {labels_dir.is_dir()}")

    return splits


def print_detection_counts(splits):
    """Count supported images and annotation files per class and split."""
    total_images = 0
    total_labels = 0

    for split_dir in splits:
        print(f"\n===== {split_dir.name} =====")

        split_images = 0
        split_labels = 0

        for class_dir in get_detection_class_dirs(split_dir):
            images_dir = class_dir / "images"
            labels_dir = class_dir / "labels"

            images = (
                [
                    path
                    for path in images_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
                ]
                if images_dir.is_dir()
                else []
            )

            labels = (
                [
                    path
                    for path in labels_dir.iterdir()
                    if path.is_file() and path.suffix.lower() == ".txt"
                ]
                if labels_dir.is_dir()
                else []
            )

            print(f"Class: {class_dir.name}")
            print(f"  Supported images: {len(images)}")
            print(f"  Label files: {len(labels)}")

            split_images += len(images)
            split_labels += len(labels)

        print(f"Total images in {split_dir.name}: {split_images}")
        print(f"Total labels in {split_dir.name}: {split_labels}")

        total_images += split_images
        total_labels += split_labels

    print("\n===== Dataset totals =====")
    print("Total supported images:", total_images)
    print("Total label files:", total_labels)


def print_detection_image_issues(splits):
    """Find unreadable images and unsupported files in image directories."""
    unreadable_images = []
    unsupported_files = []
    total_images = 0

    for split_dir in splits:
        for class_dir in get_detection_class_dirs(split_dir):
            images_dir = class_dir / "images"

            if not images_dir.is_dir():
                continue

            for path in images_dir.iterdir():
                if not path.is_file():
                    continue

                if path.suffix.lower() not in VALID_EXTENSIONS:
                    unsupported_files.append(str(path))
                    continue

                total_images += 1

                if cv2.imread(str(path)) is None:
                    unreadable_images.append(str(path))

    print("\n===== Image quality =====")
    print("Supported image files:", total_images)
    print("Unreadable images:", len(unreadable_images))
    print("Unsupported files:", len(unsupported_files))

    if unreadable_images:
        print("\nExamples of unreadable images:")
        for path in unreadable_images[:10]:
            print(path)

    if unsupported_files:
        print("\nExamples of unsupported files:")
        for path in unsupported_files[:10]:
            print(path)


def validate_image_label_matching(splits):
    """Find images without matching labels and labels without matching images."""
    missing_labels = []
    orphan_labels = []
    empty_labels = []

    for split_dir in splits:
        for class_dir in get_detection_class_dirs(split_dir):
            images_dir = class_dir / "images"
            labels_dir = class_dir / "labels"

            if not images_dir.is_dir() or not labels_dir.is_dir():
                continue

            image_paths = {
                path.stem: path
                for path in images_dir.iterdir()
                if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
            }

            label_paths = {
                path.stem: path
                for path in labels_dir.iterdir()
                if path.is_file() and path.suffix.lower() == ".txt"
            }

            for stem in sorted(image_paths.keys() - label_paths.keys()):
                missing_labels.append(str(image_paths[stem]))

            for stem in sorted(label_paths.keys() - image_paths.keys()):
                orphan_labels.append(str(label_paths[stem]))

            for label_path in label_paths.values():
                if label_path.stat().st_size == 0:
                    empty_labels.append(str(label_path))

    print("\n===== Image-label matching =====")
    print("Images without matching labels:", len(missing_labels))
    print("Labels without matching images:", len(orphan_labels))
    print("Empty label files:", len(empty_labels))

    for title, paths in [
        ("Missing labels", missing_labels),
        ("Orphan labels", orphan_labels),
        ("Empty labels", empty_labels),
    ]:
        if paths:
            print(f"\nExamples: {title}")
            for path in paths[:10]:
                print(path)


def validate_yolo_annotations(splits):
    """Validate YOLO detection annotation syntax and normalized coordinates."""
    invalid_files = []
    invalid_lines = []
    class_id_counts = Counter()
    total_boxes = 0
    empty_labels = 0

    for split_dir in splits:
        for class_dir in get_detection_class_dirs(split_dir):
            labels_dir = class_dir / "labels"

            if not labels_dir.is_dir():
                continue

            for label_path in labels_dir.iterdir():
                if not label_path.is_file() or label_path.suffix.lower() != ".txt":
                    continue

                try:
                    content = label_path.read_text(encoding="utf-8").strip()
                except (OSError, UnicodeError) as exc:
                    invalid_files.append((str(label_path), str(exc)))
                    continue

                # Empty labels are reported separately, not automatically treated as errors.
                if not content:
                    empty_labels += 1
                    continue

                for line_number, line in enumerate(content.splitlines(), start=1):
                    parts = line.split()

                    if len(parts) != 5:
                        invalid_lines.append(
                            (
                                str(label_path),
                                line_number,
                                "Expected 5 values per YOLO detection line",
                            )
                        )
                        continue

                    try:
                        class_value = float(parts[0])
                        coords = [float(value) for value in parts[1:]]
                    except ValueError:
                        invalid_lines.append(
                            (str(label_path), line_number, "Non-numeric value")
                        )
                        continue

                    if not class_value.is_integer() or class_value < 0:
                        invalid_lines.append(
                            (str(label_path), line_number, "Invalid class ID")
                        )
                        continue

                    if not all(
                        value == value and abs(value) != float("inf")
                        for value in coords
                    ):
                        invalid_lines.append(
                            (
                                str(label_path),
                                line_number,
                                "Coordinates must be finite numbers",
                            )
                        )
                        continue

                    x_center, y_center, width, height = coords

                    if not (
                        0 <= x_center <= 1
                        and 0 <= y_center <= 1
                        and 0 < width <= 1
                        and 0 < height <= 1
                    ):
                        invalid_lines.append(
                            (
                                str(label_path),
                                line_number,
                                "Coordinates or dimensions outside valid range",
                            )
                        )
                        continue

                    if (
                        x_center - width / 2 < 0
                        or x_center + width / 2 > 1
                        or y_center - height / 2 < 0
                        or y_center + height / 2 > 1
                    ):
                        invalid_lines.append(
                            (
                                str(label_path),
                                line_number,
                                "Bounding box extends outside image boundaries",
                            )
                        )
                        continue

                    class_id = int(class_value)
                    class_id_counts[class_id] += 1
                    total_boxes += 1

    print("\n===== YOLO annotation validation =====")
    print("Valid bounding boxes:", total_boxes)
    print("Empty label files:", empty_labels)
    print("Unreadable label files:", len(invalid_files))
    print("Invalid annotation lines:", len(invalid_lines))
    print("Bounding boxes per class ID:", dict(sorted(class_id_counts.items())))

    if invalid_files:
        print("\nExamples of unreadable label files:")
        for path, error in invalid_files[:10]:
            print(path, "-", error)

    if invalid_lines:
        print("\nExamples of invalid annotation lines:")
        for path, line_number, reason in invalid_lines[:10]:
            print(f"{path}:{line_number} - {reason}")


def collect_detection_image_shapes(splits):
    """Count image dimensions across all detection splits and classes."""
    image_shapes = Counter()

    for split_dir in splits:
        for class_dir in get_detection_class_dirs(split_dir):
            images_dir = class_dir / "images"

            if not images_dir.is_dir():
                continue

            for image_path in images_dir.iterdir():
                if (
                    not image_path.is_file()
                    or image_path.suffix.lower() not in VALID_EXTENSIONS
                ):
                    continue

                image = cv2.imread(str(image_path))

                if image is not None:
                    height, width = image.shape[:2]
                    image_shapes[(height, width)] += 1

    return image_shapes


def print_detection_image_dimensions(image_shapes):
    """Print common image dimensions and dimension ranges."""
    print("\n===== Image dimensions =====")

    if not image_shapes:
        print("No readable images found.")
        return

    print("Number of unique dimensions:", len(image_shapes))
    print("\nMost common dimensions:")

    for (height, width), count in image_shapes.most_common(10):
        print(f"{width} × {height}: {count} images")

    dimensions = list(image_shapes)

    print("\nMinimum width:", min(width for height, width in dimensions))
    print("Maximum width:", max(width for height, width in dimensions))
    print("Minimum height:", min(height for height, width in dimensions))
    print("Maximum height:", max(height for height, width in dimensions))


def plot_detection_dimension_counts(image_shapes):
    """Plot the 10 most common image dimensions."""
    if not image_shapes:
        print("No image dimensions available to plot.")
        return

    top_shapes = image_shapes.most_common(10)

    labels = [f"{width} × {height}" for (height, width), count in top_shapes]
    counts = [count for shape, count in top_shapes]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, counts)
    plt.title("Top 10 Most Common Detection Image Dimensions")
    plt.xlabel("Image dimensions (width × height)")
    plt.ylabel("Number of images")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_detection_aspect_ratios(image_shapes):
    """Plot the image aspect ratio distribution."""
    if not image_shapes:
        print("No image dimensions available to plot.")
        return

    aspect_ratios = []

    for (height, width), count in image_shapes.items():
        aspect_ratios.extend([width / height] * count)

    plt.figure(figsize=(9, 4))
    plt.hist(aspect_ratios, bins=30)
    plt.title("Detection Image Aspect Ratio Distribution")
    plt.xlabel("Width / Height")
    plt.ylabel("Number of images")
    plt.tight_layout()
    plt.show()

    print("Minimum aspect ratio:", round(min(aspect_ratios), 3))
    print("Maximum aspect ratio:", round(max(aspect_ratios), 3))


def display_detection_samples(splits):
    """Display random images with their YOLO bounding boxes."""
    random.seed(RANDOM_STATE)

    num_samples = 2

    for split_dir in splits:
        class_dirs = get_detection_class_dirs(split_dir)

        if not class_dirs:
            continue

        fig, axes = plt.subplots(
            nrows=len(class_dirs),
            ncols=num_samples,
            figsize=(12, 4 * len(class_dirs)),
            squeeze=False,
        )

        for row, class_dir in enumerate(class_dirs):
            images_dir = class_dir / "images"
            labels_dir = class_dir / "labels"

            if not images_dir.is_dir() or not labels_dir.is_dir():
                for col in range(num_samples):
                    axes[row, col].axis("off")
                continue

            image_paths = [
                path
                for path in images_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in VALID_EXTENSIONS
                and (labels_dir / f"{path.stem}.txt").is_file()
                and cv2.imread(str(path)) is not None
            ]

            selected_paths = random.sample(
                image_paths,
                k=min(num_samples, len(image_paths)),
            )

            for col in range(num_samples):
                ax = axes[row, col]

                image_path = selected_paths[col]
                label_path = labels_dir / f"{image_path.stem}.txt"

                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                height, width = image.shape[:2]

                ax.imshow(image)
                ax.set_title(f"{split_dir.name} / {class_dir.name}\n{image_path.name}")
                ax.axis("off")

                content = label_path.read_text(encoding="utf-8").strip()

                if not content:
                    ax.set_title(
                        f"{split_dir.name} / {class_dir.name}\n"
                        f"{image_path.name}\nNo bounding boxes"
                    )
                    continue

                for line in content.splitlines():
                    parts = line.split()

                    if len(parts) != 5:
                        continue

                    try:
                        class_value = float(parts[0])
                        x_center, y_center, box_width, box_height = map(
                            float, parts[1:]
                        )
                    except ValueError:
                        continue

                    if (
                        not class_value.is_integer()
                        or not all(
                            value == value and abs(value) != float("inf")
                            for value in (x_center, y_center, box_width, box_height)
                        )
                        or not (
                            0 <= x_center <= 1
                            and 0 <= y_center <= 1
                            and 0 < box_width <= 1
                            and 0 < box_height <= 1
                        )
                        or x_center - box_width / 2 < 0
                        or x_center + box_width / 2 > 1
                        or y_center - box_height / 2 < 0
                        or y_center + box_height / 2 > 1
                    ):
                        continue

                    x1 = (x_center - box_width / 2) * width
                    y1 = (y_center - box_height / 2) * height
                    box_w = box_width * width
                    box_h = box_height * height

                    rectangle = plt.Rectangle(
                        (x1, y1),
                        box_w,
                        box_h,
                        fill=False,
                        edgecolor="red",
                        linewidth=2,
                    )

                    ax.add_patch(rectangle)
                    ax.text(
                        x1,
                        max(0, y1 - 5),
                        f"ID: {int(class_value)}",
                        color="white",
                        fontsize=8,
                        bbox={"facecolor": "black", "alpha": 0.7},
                    )

        fig.suptitle(f"Detection Samples — {split_dir.name}", fontsize=14)
        plt.tight_layout()
        plt.show()