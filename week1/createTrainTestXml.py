import sys, os, random

# XML backend: prefer lxml, else fallback to stdlib
try:
    from lxml import etree as ET
    PRETTY = dict(pretty_print=True)
except ImportError:
    import xml.etree.ElementTree as ET
    PRETTY = {}  # stdlib ET has no pretty_print
    print("lxml not found; using stdlib xml (no pretty-print). To install: pip install lxml")

def createXml(imageNames, xmlPath, numPoints, dataDir):
    # Root
    dataset = ET.Element('dataset')
    ET.SubElement(dataset, 'name').text = 'Training Faces'
    images = ET.SubElement(dataset, 'images')

    numFiles = len(imageNames)
    print(f'{xmlPath} : {numFiles} files')

    for k, imageName in enumerate(imageNames, 1):
        print(f'{k}:{numFiles} - {imageName}')

        # Build paths
        img_path = os.path.join(dataDir, imageName)
        rect_path = os.path.join(dataDir, os.path.splitext(imageName)[0] + '_rect.txt')
        points_path = os.path.join(dataDir, os.path.splitext(imageName)[0] + f'_bv{numPoints}.txt')

        # Basic checks
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Missing image: {img_path}")
        if not os.path.isfile(rect_path):
            raise FileNotFoundError(f"Missing rect file: {rect_path}")
        if not os.path.isfile(points_path):
            raise FileNotFoundError(f"Missing points file: {points_path}")

        # Read rect
        with open(rect_path, 'r') as f:
            rect_vals = f.readline().split()
        if len(rect_vals) < 4:
            raise ValueError(f"Bad rect line in {rect_path}: {rect_vals}")
        left, top, width, height = [str(int(float(v))) for v in rect_vals[:4]]

        # XML nodes
        image = ET.SubElement(images, 'image', file=img_path)  # use full path; or make relative
        box = ET.SubElement(image, 'box', top=top, left=left, width=width, height=height)

        # Read points
        with open(points_path, 'r') as f:
            for i, line in enumerate(f):
                xs, ys = line.split()[:2]
                x = str(int(float(xs)))
                y = str(int(float(ys)))
                name = str(i).zfill(2)
                ET.SubElement(box, 'part', name=name, x=x, y=y)

    # Write XML
    tree = ET.ElementTree(dataset)
    print(f'writing on disk: {xmlPath}')
    if hasattr(tree, 'write'):
        tree.write(xmlPath, xml_declaration=True, encoding='UTF-8', **PRETTY)
    else:
        # stdlib ElementTree (older Python) fallback
        ET.ElementTree(dataset).write(xmlPath, xml_declaration=True, encoding='UTF-8')

if __name__ == '__main__':
    # Defaults
    fldDatadir = "./data/facial_landmark_data"
    numPoints = 70

    if len(sys.argv) >= 2:
        fldDatadir = sys.argv[1]
    if len(sys.argv) >= 3:
        numPoints = int(sys.argv[2])  # keep as int

    # Read image list
    with open(os.path.join(fldDatadir, 'image_names.txt'), 'r') as d:
        imageNames = [x.strip() for x in d if x.strip()]

    # Optional: limit dataset size
    # n = 1000
    # imageNames = random.sample(imageNames, min(n, len(imageNames)))

    # Reproducible split
    random.seed(42)
    total = len(imageNames)
    numTest = max(1, int(0.05 * total))  # at least 1
    testFiles = set(random.sample(imageNames, numTest))
    trainFiles = [n for n in imageNames if n not in testFiles]  # keep order
    testFiles = [n for n in imageNames if n in testFiles]

    # Generate XMLs
    createXml(trainFiles, os.path.join(fldDatadir, 'training_with_face_landmarks.xml'), numPoints, fldDatadir)
    createXml(testFiles,  os.path.join(fldDatadir, 'testing_with_face_landmarks.xml'),  numPoints, fldDatadir)
