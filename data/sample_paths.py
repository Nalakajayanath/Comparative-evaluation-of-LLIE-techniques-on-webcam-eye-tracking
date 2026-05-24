import os


def build_eye_image_rel_path(image_rel, eye):
    """Map eval-list path dayXX/0001.jpg -> dayXX/left_0001.jpg."""
    day_dir, filename = os.path.split(image_rel.replace("\\", "/"))
    stem, ext = os.path.splitext(filename)
    return os.path.normpath(os.path.join(day_dir, f"{eye}_{stem}{ext}"))


def parse_sample_path(rel_path):
    """
    Parse p00/day01/left_0001.jpg into subject, day, eye, frame_index.
    Supports legacy paths without an eye prefix (eye=None).
    """
    parts = os.path.normpath(rel_path).split(os.sep)
    subject = parts[-3]
    day = parts[-2]
    filename = parts[-1]
    stem, _ = os.path.splitext(filename)

    if stem.startswith("left_"):
        eye = "left"
        frame_index = int(stem[5:]) - 1
    elif stem.startswith("right_"):
        eye = "right"
        frame_index = int(stem[6:]) - 1
    else:
        eye = None
        frame_index = int(stem) - 1

    return subject, day, eye, frame_index
