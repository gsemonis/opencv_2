"""
Public domain / Boost-style note as in original
"""
import os
import sys
import dlib

print("USAGE : python trainFLD.py <path to facial_landmark_data folder> <number of points>")

# Defaults
fldDatadir = "./data/facial_landmark_data"
numPoints = 70

if len(sys.argv) >= 2:
    fldDatadir = sys.argv[1]
if len(sys.argv) >= 3:
    # keep as int internally
    try:
        numPoints = int(sys.argv[2])
    except ValueError:
        raise SystemExit("numPoints must be an integer (e.g. 68, 70)")

modelName = f"shape_predictor_{numPoints}_face_landmarks.dat"

# ---- Training options ----
options = dlib.shape_predictor_training_options()
options.be_verbose = True
options.num_threads = max(1, os.cpu_count() or 1)

# Tree/forest params (dlib defaults are reasonable; these are typical)
options.cascade_depth = 10
options.num_trees_per_cascade_level = 500
options.tree_depth = 4

# Regularization / sampling
options.nu = 0.1
options.lambda_param = 0.1
options.num_test_splits = 20
options.oversampling_amount = 20
options.oversampling_translation_jitter = 0.1  # small shift jitter helps
# Stop early if no improvement (in training objective)
#options.epsilon = 1e-7

# Optional: reproducibility
options.random_seed = "42"

trainingXmlPath = os.path.join(fldDatadir, "training_with_face_landmarks.xml")
testingXmlPath  = os.path.join(fldDatadir, "testing_with_face_landmarks.xml")
outputModelPath = os.path.join(fldDatadir, modelName)

def require_nonempty_xml(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing XML: {path}")
    # quick sanity check: file has at least one <image>
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        if "<image " not in f.read():
            raise RuntimeError(f"No <image> entries found in {path}")

try:
    require_nonempty_xml(trainingXmlPath)
    require_nonempty_xml(testingXmlPath)

    print(f"Training → {outputModelPath}")
    dlib.train_shape_predictor(trainingXmlPath, outputModelPath, options)

    # Evaluate
    train_err = dlib.test_shape_predictor(trainingXmlPath, outputModelPath)
    test_err  = dlib.test_shape_predictor(testingXmlPath,  outputModelPath)
    print(f"\nTraining accuracy (mean error in pixels): {train_err:.6f}")
    print(f"Testing  accuracy (mean error in pixels): {test_err:.6f}")

except Exception as e:
    print("Error during training/testing:", e)
    print("Check your XML paths and annotation format.")
