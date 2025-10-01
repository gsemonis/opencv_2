#!/usr/bin/env python3
# Public domain / Boost-style notice as in original

import os
import sys
import random
import argparse

# Prefer lxml for nice pretty-print; fall back to stdlib
try:
    from lxml import etree as ET
    PRETTY = dict(pretty_print=True)
    HAS_LXML = True
except ImportError:
    import xml.etree.ElementTree as ET
    PRETTY = {}  # stdlib ET lacks pretty_print
    HAS_LXML = False
    print("lxml not found; using stdlib xml (no pretty-print). To install: pip install lxml")

# Optional: bounds validation if OpenCV is available
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


def clean_lines(fp):
    """Yield non-empty, non-comment lines."""
    for line in fp:
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        yield s


def read_rect(rect_path):
    """Read left, top, width, height (as ints) from a rect file's first non-empty line."""
    with open(rect_path, 'r', encoding='utf-8', errors='strict') as f:
        line = next(clean_lines(f), None)
    if not line:
        raise ValueError(f"No rect data in {rect_path}")
    vals = line.split()
    if len(vals) < 4:
        raise ValueError(f"Bad rect line in {rect_path}: {vals}")
    l, t, w, h = [int(float(v)) for v in vals[:4]]
    return l, t, w, h


def read_points(points_path, num_points):
    """Read exactly num_points (x, y) pairs from points file. Returns list of (int_x, int_y)."""
    with open(points_path, 'r', encoding='utf-8', errors='strict') as f:
        pts = []
        for s in clean_lines(f):
            xs, ys = s.split()[:2]
            pts.append((int(float(xs)), int(float(ys))))
    if len(pts) != num_points:
        raise ValueError(f"{points_path}: expected {num_points} points, got {len(pts)}")
    return pts


def bounds_check(img_path, rect, points):
    """Validate that rect and points lie within the image bounds (if OpenCV is present)."""
    if not HAS_CV2:
        return  # skip silently if cv2 not available
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    H, W = img.shape[:2]
    l, t, w, h = rect
    if not (0 <= l < W and 0 <= t < H and w > 0 and h > 0 and l + w <= W and t + h <= H):
        raise ValueError(f"Rect out of bounds for {img_path}: {rect} vs image {W}x{H}")
    for (x, y) in points:
        if not (0 <= x < W and 0 <= y < H):
            raise ValueError(f"Point {(x, y)} out of bounds for {img_path} ({W}x{H})")


def create_xml(image_names, xml_path, num_points, data_dir, use_abs_paths=False, do_bounds_check=False):
    """Create a dlib-compatible XML at xml_path for the given images."""
    # Root
    dataset = ET.Element('dataset')
    ET.SubElement(dataset, 'name').text = 'Training Faces'
    images = ET.SubElement(dataset, 'images')

    num_files = len(image_names)
    print(f'{xml_path} : {num_files} files')

    xml_dir = os.path.dirname(os.path.abspath(xml_path))

    for k, image_name in enumerate(image_names, 1):
        print(f'{k}:{num_files} - {image_name}')

        # Build paths
        img_path = os.path.join(data_dir, image_name)
        base = os.path.splitext(image_name)[0]
        rect_path = os.path.join(data_dir, base + '_rect.txt')
        # points filename pattern: <image>_bv{num_points}.txt
        points_path = os.path.join(data_dir, base + f'_bv{num_points}.txt')

        # Basic checks
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Missing image: {img_path}")
        if not os.path.isfile(rect_path):
            raise FileNotFoundError(f"Missing rect file: {rect_path}")
        if not os.path.isfile(points_path):
            raise FileNotFoundError(f"Missing points file: {points_path}")

        # Read annotation
        l, t, w, h = read_rect(rect_path)
        pts = read_points(points_path, num_points)

        # Optional: bounds validation
        if do_bounds_check:
            bounds_check(img_path, (l, t, w, h), pts)

        # Decide path style for XML
        if use_abs_paths:
            file_attr = os.path.abspath(img_path)
        else:
            rel = os.path.relpath(os.path.abspath(img_path), start=xml_dir)
            file_attr = rel.replace('\\', '/')  # normalize for portability

        # XML nodes
        image = ET.SubElement(images, 'image', file=file_attr)
        box = ET.SubElement(image, 'box',
                           top=str(t), left=str(l), width=str(w), height=str(h))
        for i, (x, y) in enumerate(pts):
            ET.SubElement(box, 'part', name=str(i).zfill(2), x=str(x), y=str(y))

    # Write XML
    tree = ET.ElementTree(dataset)
    # Pretty indent if using stdlib on Python 3.9+
    if not HAS_LXML:
        try:
            ET.indent(tree, space="  ")  # type: ignore[attr-defined]
        except Exception:
            pass

    os.makedirs(os.path.dirname(os.path.abspath(xml_path)), exist_ok=True)
    print(f'writing on disk: {xml_path}')
    tree.write(xml_path, xml_declaration=True, encoding='UTF-8', **PRETTY)


def main():
    parser = argparse.ArgumentParser(
        description="Generate dlib training/testing XMLs from facial landmark annotations."
    )
    parser.add_argument("data_dir", nargs='?', default="./data/facial_landmark_data",
                        help="Path to facial_landmark_data folder (default: ./data/facial_landmark_data)")
    parser.add_argument("num_points", nargs='?', type=int, default=70,
                        help="Number of landmarks/points per face (default: 70)")
    parser.add_argument("--test-split", type=float, default=0.05,
                        help="Fraction of images for testing (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible split (default: 42)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on total images used (debugging)")
    parser.add_argument("--abs-paths", action="store_true",
                        help="Store absolute image paths in XML (default: relative)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate rects/points are within image bounds (requires OpenCV)")

    args = parser.parse_args()

    data_dir = args.data_dir
    num_points = args.num_points

    image_names_path = os.path.join(data_dir, 'image_names.txt')
    if not os.path.isfile(image_names_path):
        print("image_names.txt not found at:", image_names_path)
        sys.exit(1)

    with open(image_names_path, 'r', encoding='utf-8', errors='strict') as d:
        image_names = [x.strip() for x in d if x.strip()]

    if args.limit is not None:
        image_names = image_names[:max(0, args.limit)]

    total = len(image_names)
    if total == 0:
        print("No images listed in image_names.txt")
        sys.exit(1)

    random.seed(args.seed)
    num_test = max(1, int(args.test_split * total))
    test_set = set(random.sample(image_names, num_test))

    # Keep original order
    train_files = [n for n in image_names if n not in test_set]
    test_files = [n for n in image_names if n in test_set]

    train_xml = os.path.join(data_dir, 'training_with_face_landmarks.xml')
    test_xml = os.path.join(data_dir, 'testing_with_face_landmarks.xml')

    create_xml(train_files, train_xml, num_points, data_dir,
               use_abs_paths=args.abs_paths, do_bounds_check=args.validate)
    create_xml(test_files, test_xml, num_points, data_dir,
               use_abs_paths=args.abs_paths, do_bounds_check=args.validate)

    print("Done.")


if __name__ == '__main__':
    main()
