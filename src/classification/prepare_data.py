import hashlib
import re
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

# ── Constants ────────────────────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.15
RANDOM_STATE = 42
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
EXPECTED_CLASSES = {"glioma", "meningioma", "notumor", "pituitary"}


# ── 1. Project paths ────────────────────────────────────────────────────────
def get_project_paths(project_root=None):
    """Return a dict of paths used by the preparation pipeline."""
    root = (
        Path(project_root).resolve()
        if project_root
        else Path(__file__).resolve().parents[2]
    )
    if (root / "Data" / "origine" / "classification").is_dir():
        original = root / "Data" / "origine" / "classification"
    elif (root / "Data" / "classification").is_dir():
        original = root / "Data" / "classification"
    else:
        raise FileNotFoundError(f"Classification dataset not found under: {root / 'Data'}")

    if (root / "Data" / "clean").is_dir():
        prepared = root / "Data" / "clean" / "classification_prepared"
    else:
        prepared = root / "Data" / "classification_prepared"

    return {
        "project_root": root,
        "original_data_dir": original,
        "prepared_data_dir": prepared,
        "train_pool_dir": prepared / "train_pool",
        "test_dir": prepared / "test",
    }


def print_configuration(paths):
    """Display preparation settings and seed RNGs for reproducibility."""
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    print(f"Project root:      {paths['project_root']}")
    print(f"Original dataset:  {paths['original_data_dir']}")
    print(f"Prepared dataset:  {paths['prepared_data_dir']}")
    print(f"Image size:        {IMAGE_SIZE}")
    print(f"Batch size:        {BATCH_SIZE}")
    print(f"Validation split:  {VALIDATION_SPLIT}")


# ── 2. Hashing helpers ──────────────────────────────────────────────────────
def _sha256(file_path):
    """Return the SHA-256 hex digest of *file_path*."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── 3. Duplicate detection & removal ────────────────────────────────────────
def analyze_duplicates(dataset_dir):
    """Find and report groups of identical images (by SHA-256).

    Returns
    -------
    dict[str, list[Path]]
        Mapping from hash → list of paths for groups with >1 file.
    """
    hash_groups = defaultdict(list)
    for p in dataset_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            hash_groups[_sha256(p)].append(p)

    duplicates = {h: paths for h, paths in hash_groups.items() if len(paths) > 1}

    print(f"Duplicate groups: {len(duplicates)}")
    print(
        f"Files in duplicate groups: "
        f"{sum(len(v) for v in duplicates.values())}"
    )
    for i, (digest, paths) in enumerate(list(duplicates.items())[:10], 1):
        print(f"\n  Group {i} | SHA-256: {digest[:16]}…")
        for p in paths:
            print(f"    {p.relative_to(dataset_dir)}")

    return duplicates


def remove_duplicates(image_paths):
    """De-duplicate image paths by SHA-256 hash in memory.

    Does NOT modify or delete any files from the original dataset.
    Keeps one deterministic copy of each unique image (prioritizing 'Tr-'
    training images over 'Te-' test images to preserve the training pool and
    prevent train/test leakage).

    Parameters
    ----------
    image_paths : iterable of (str | Path)
        List of image file paths to de-duplicate.

    Returns
    -------
    tuple[list[str], list[Path]]
        (unique_hashes, unique_image_paths)
    """
    sorted_paths = sorted(
        [Path(p) for p in image_paths if Path(p).is_file()],
        key=lambda p: (0 if p.name.lower().startswith("tr") else 1, p.name),
    )

    seen = set()
    unique_hashes, unique_images = [], []

    for path in sorted_paths:
        h = _sha256(path)
        if h in seen:
            continue

        seen.add(h)
        unique_hashes.append(h)
        unique_images.append(path)

    return unique_hashes, unique_images


# ── 4. Validation ───────────────────────────────────────────────────────────
def _get_split(image_path):
    """Return 'train_pool' or 'test' based on the Tr-/Te- filename prefix."""
    m = re.match(r"^(Tr|Te)(?:[-_ ]|$)", image_path.name, re.IGNORECASE)
    if m is None:
        return None
    return "train_pool" if m.group(1).lower() == "tr" else "test"


def _is_readable(image_path):
    """Return True if the image can be opened and verified by Pillow."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except (OSError, ValueError, UnidentifiedImageError):
        return False


def validate_files(original_data_dir, image_paths=None):
    """Validate images and return a manifest of valid entries.

    Each manifest entry is a dict with keys:
        source, class_name, split, sha256
    """
    class_dirs = sorted(
        p for p in original_data_dir.iterdir() if p.is_dir()
    )
    found = {p.name.lower() for p in class_dirs}
    if found != EXPECTED_CLASSES:
        raise ValueError(
            f"Unexpected classes.\n"
            f"  Expected: {sorted(EXPECTED_CLASSES)}\n"
            f"  Found:    {sorted(found)}"
        )

    manifest = []
    seen_hashes = set()
    unsupported, unreadable, bad_prefix = [], [], []
    duplicates_skipped = 0

    if image_paths is not None:
        candidate_images = sorted(
            [Path(p) for p in image_paths if Path(p).is_file()],
            key=lambda p: (0 if p.name.lower().startswith("tr") else 1, p.name),
        )
    else:
        candidate_images = sorted(
            [
                img
                for class_dir in class_dirs
                for img in class_dir.iterdir()
                if img.is_file()
            ],
            key=lambda p: (0 if p.name.lower().startswith("tr") else 1, p.name),
        )

    for img in candidate_images:
        if img.suffix.lower() not in SUPPORTED_EXTENSIONS:
            unsupported.append(img)
            continue
        split = _get_split(img)
        if split is None:
            bad_prefix.append(img)
            continue
        if not _is_readable(img):
            unreadable.append(img)
            continue

        h = _sha256(img)
        if h in seen_hashes:
            duplicates_skipped += 1
            continue
        seen_hashes.add(h)

        manifest.append({
            "source": img,
            "class_name": img.parent.name.lower(),
            "split": split,
            "sha256": h,
        })

    print(f"Valid images:                {len(manifest)}")
    print(f"Unsupported files:           {len(unsupported)}")
    print(f"Unreadable images:           {len(unreadable)}")
    print(f"Unrecognized prefixes:       {len(bad_prefix)}")
    if duplicates_skipped:
        print(f"Duplicate images skipped:    {duplicates_skipped}")

    if bad_prefix:
        raise ValueError("Some images lack a recognizable Tr-/Te- prefix.")
    if not manifest:
        raise ValueError("No valid images found.")

    # Cross-split / cross-class leakage check
    hash_splits = defaultdict(set)
    hash_classes = defaultdict(set)
    for item in manifest:
        hash_splits[item["sha256"]].add(item["split"])
        hash_classes[item["sha256"]].add(item["class_name"])

    print(
        f"Hashes spanning train/test:  "
        f"{sum(len(s) > 1 for s in hash_splits.values())}"
    )
    print(
        f"Hashes spanning classes:     "
        f"{sum(len(c) > 1 for c in hash_classes.values())}"
    )
    return manifest


# ── 5. Organize prepared data ───────────────────────────────────────────────
def organize_prepared_data(manifest, prepared_data_dir):
    """Copy validated images into train_pool/ and test/ subdirectories."""
    if prepared_data_dir.exists():
        shutil.rmtree(prepared_data_dir)

    for split in ("train_pool", "test"):
        for cls in sorted(EXPECTED_CLASSES):
            (prepared_data_dir / split / cls).mkdir(parents=True, exist_ok=True)

    for item in manifest:
        dst = prepared_data_dir / item["split"] / item["class_name"]
        shutil.copy2(item["source"], dst / item["source"].name)

    print("Prepared directories created successfully.")


# ── 6. Load datasets ────────────────────────────────────────────────────────
def load_datasets(train_pool_dir, test_dir):
    """Load training, validation and test datasets using Keras utilities.

    Returns
    -------
    tuple
        (train_raw, val_raw, test_raw, class_names, class_mapping)
    """
    def _load(directory, subset=None, shuffle=True):
        return tf.keras.utils.image_dataset_from_directory(
            directory,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="int",
            validation_split=VALIDATION_SPLIT if subset else None,
            subset=subset,
            seed=RANDOM_STATE if subset else None,
            shuffle=shuffle,
            pad_to_aspect_ratio=True,
        )

    train_raw = _load(train_pool_dir, subset="training")
    val_raw = _load(train_pool_dir, subset="validation")
    test_raw = _load(test_dir, shuffle=False)

    class_names = train_raw.class_names
    assert class_names == val_raw.class_names == test_raw.class_names
    assert set(class_names) == EXPECTED_CLASSES

    class_mapping = dict(enumerate(class_names))
    print(f"Class-to-index mapping: {class_mapping}")
    return train_raw, val_raw, test_raw, class_names, class_mapping


# ── 7. Preprocessing ────────────────────────────────────────────────────────
def _normalize(image, label):
    """Cast to float32 and scale pixels to [0, 1]."""
    return tf.cast(image, tf.float32) / 255.0, label


def preprocess_datasets(train_raw, val_raw, test_raw):
    """Apply normalization to all three datasets."""
    autotune = tf.data.AUTOTUNE
    return tuple(
        ds.map(_normalize, num_parallel_calls=autotune)
        for ds in (train_raw, val_raw, test_raw)
    )


# ── 8. Visualization ────────────────────────────────────────────────────────
def visualize_preprocessed_images(train_ds, class_names):
    """Display the first 9 images from one training batch."""
    for images, labels in train_ds.take(1):
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image dtype:  {images.dtype}")
        print(f"Pixel range:  [{float(tf.reduce_min(images)):.1f}, "
              f"{float(tf.reduce_max(images)):.1f}]")

        n = min(9, int(images.shape[0]))
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        for ax in axes.flatten():
            ax.axis("off")
        for i in range(n):
            axes.flat[i].imshow(
                tf.clip_by_value(images[i], 0.0, 1.0).numpy()
            )
            axes.flat[i].set_title(
                f"Label: {class_names[int(labels[i].numpy())]}"
            )
        fig.suptitle("Preprocessed Training Images")
        plt.tight_layout()
        plt.show()


# ── 9. Augmentation ─────────────────────────────────────────────────────────
def _random_gaussian_blur(images, probability=0.15):
    """Apply a 3×3 Gaussian blur to the batch with given probability."""
    k1d = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    k1d /= k1d.sum()
    k2d = np.outer(k1d, k1d)
    kernel = tf.constant(
        np.tile(k2d[:, :, None, None], (1, 1, 3, 1)),
        dtype=tf.float32,
    )

    def _blur():
        return tf.nn.depthwise_conv2d(
            images, kernel, strides=[1, 1, 1, 1], padding="SAME"
        )

    return tf.cond(tf.random.uniform(()) < probability, _blur, lambda: images)


def augment_datasets(train_ds, val_ds, test_ds):
    """Apply training-only augmentation and prefetch all datasets."""
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(mode="horizontal", seed=RANDOM_STATE),
            tf.keras.layers.GaussianNoise(stddev=0.02, seed=RANDOM_STATE),
        ],
        name="training_augmentation",
    )

    def _augment(images, labels):
        images = augmentation(images, training=True)
        images = _random_gaussian_blur(images)
        return tf.clip_by_value(images, 0.0, 1.0), labels

    autotune = tf.data.AUTOTUNE
    return (
        train_ds.map(_augment, num_parallel_calls=autotune).prefetch(autotune),
        val_ds.prefetch(autotune),
        test_ds.prefetch(autotune),
    )


# ── 10. Final validation ────────────────────────────────────────────────────
def _count_examples(dataset):
    """Count the total number of examples in a batched dataset."""
    return sum(int(tf.shape(labels)[0]) for _, labels in dataset)


def validate_dataset_readiness(train_raw, val_raw, test_raw, val_ds, class_names):
    """Run sanity checks on shapes, dtypes, pixel ranges and labels."""
    counts = {
        "train": _count_examples(train_raw),
        "validation": _count_examples(val_raw),
        "test": _count_examples(test_raw),
    }
    for name, n in counts.items():
        print(f"{name.capitalize()} images: {n}")

    for images, labels in val_ds.take(1):
        assert images.shape[1:] == (*IMAGE_SIZE, 3)
        assert images.dtype == tf.float32
        assert labels.dtype.is_integer
        assert float(tf.reduce_min(images)) >= 0.0
        assert float(tf.reduce_max(images)) <= 1.0
        assert int(tf.reduce_min(labels)) >= 0
        assert int(tf.reduce_max(labels)) < len(class_names)

    assert all(counts.values())
    print("All dataset readiness checks passed.")
    return counts