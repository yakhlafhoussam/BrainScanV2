import cv2
import random
import matplotlib.pyplot as plt
import hashlib

from PIL import Image

from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "Data" / "origine"

IMAGE_SIZE = (224, 224)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

RANDOM_STATE = 42


def overview():
    print("Project directory:", PROJECT_DIR)
    print("Dataset directory:", DATA_DIR)
    print("Dataset exists:", DATA_DIR.exists())
    print("Image size:", IMAGE_SIZE)
    print("Supported extensions:", VALID_EXTENSIONS)


def exploration():
    paths = sorted(folder for folder in DATA_DIR.iterdir() if folder.is_dir())
    print("Number of Paths in Data folder:", len(paths))

    data = [folder.name for folder in paths]
    for d in data:
        print(d)

    return paths[0], paths[1]


def get_class_dirs(CPATH: Path):
    return sorted(folder for folder in CPATH.iterdir() if folder.is_dir())


def get_duplicate_images(classification_dir: Path = None):
    """Return groups of classification images with identical file contents."""
    if classification_dir is None:
        classification_dir = (
            DATA_DIR / "classification"
            if (DATA_DIR / "classification").is_dir()
            else DATA_DIR
        )
    image_hashes = {}

    if not classification_dir.is_dir():
        return {}

    for class_dir in get_class_dirs(classification_dir):
        for image_path in sorted(class_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            image_hashes.setdefault(digest, []).append(image_path)

    return {digest: paths for digest, paths in image_hashes.items() if len(paths) > 1}


def print_class_summary(class_dirs):
    print("Number of classes:", len(class_dirs))

    count = 0
    counts = 0

    for folder in class_dirs:
        files = [file for file in folder.iterdir() if file.is_file()]
        images = [file for file in files if file.suffix.lower() in VALID_EXTENSIONS]
        prefixes = Counter(
            file.name[:3] for file in images if file.name.startswith(("Tr-", "Te-"))
        )

        counts += len(files)
        count += len(images)

        print(f"Class: {folder.name}")
        print(f"  Total files: {len(files)}")
        print(f"  Supported images: {len(images)}")
        print(f"  Training prefix (Tr-): {prefixes['Tr-']}")
        print(f"  Testing prefix (Te-): {prefixes['Te-']}")
        print()

    print(f"Total of images : {count}\nFrom {counts} of files")


def print_image_issues(class_dirs):
    corrupted_images = []
    unsupported_files = []
    total_images = 0

    for folder in class_dirs:
        for file in folder.iterdir():
            if not file.is_file():
                continue

            if file.suffix.lower() not in VALID_EXTENSIONS:
                unsupported_files.append(str(file))
                continue

            total_images += 1
            if cv2.imread(str(file)) is None:
                corrupted_images.append(str(file))

    print("Total supported image files:", total_images)
    print("Unreadable images:", len(corrupted_images))
    print("Unsupported files:", len(unsupported_files))

    if corrupted_images:
        print("\nExamples of unreadable images:")
        for path in corrupted_images[:10]:
            print(path)

    if unsupported_files:
        print("\nExamples of unsupported files:")
        for path in unsupported_files[:10]:
            print(path)


def display_sample_images(class_dirs):
    random.seed(RANDOM_STATE)
    num_samples = 3
    _, axes = plt.subplots(nrows=len(class_dirs), ncols=num_samples, figsize=(12, 12))

    for row, folder in enumerate(class_dirs):
        image_paths = [
            file
            for file in folder.iterdir()
            if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS
        ]
        selected_paths = random.sample(
            image_paths, k=min(num_samples, len(image_paths))
        )

        for col, image_path in enumerate(selected_paths):
            image = cv2.imread(str(image_path))
            axes[row, col].imshow(image)
            axes[row, col].set_title(f"{folder.name}\n{image_path.name}", fontsize=8)
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()


def collect_image_shapes(class_dirs):
    image_shapes = Counter()

    for folder in class_dirs:
        for image_path in folder.iterdir():
            image = cv2.imread(str(image_path))
            if image is not None:
                height, width = image.shape[:2]
                image_shapes[(height, width)] += 1

    return image_shapes


def print_image_dimensions(image_shapes):
    print("Number of unique image dimensions:", len(image_shapes))
    print("\nMost common dimensions:")

    for shape, count in image_shapes.most_common(10):
        print(f"{shape}: {count} images")

    print("\nMinimum width:", min(w for h, w in image_shapes))
    print("Maximum width:", max(w for h, w in image_shapes))
    print("Minimum height:", min(h for h, w in image_shapes))
    print("Maximum height:", max(h for h, w in image_shapes))


def plot_dimension_counts(image_shapes):
    top_shapes = image_shapes.most_common(10)
    shape_labels = [f"{width} × {height}" for (height, width), count in top_shapes]
    shape_counts = [count for shape, count in top_shapes]

    plt.figure(figsize=(10, 5))
    plt.bar(shape_labels, shape_counts)
    plt.title("Top 10 Most Common Image Dimensions")
    plt.xlabel("Image dimensions (width × height)")
    plt.ylabel("Number of images")
    plt.tight_layout()
    plt.show()


def plot_aspect_ratios(image_shapes):
    aspect_ratios = []
    for (height, width), count in image_shapes.items():
        aspect_ratios.extend([width / height] * count)

    plt.figure(figsize=(9, 4))
    plt.hist(aspect_ratios, bins=30)
    plt.title("Distribution of Image Aspect Ratios")
    plt.xlabel("Width / Height")
    plt.ylabel("Number of images")
    plt.tight_layout()
    plt.show()

    print("Minimum aspect ratio:", round(min(aspect_ratios), 3))
    print("Maximum aspect ratio:", round(max(aspect_ratios), 3))