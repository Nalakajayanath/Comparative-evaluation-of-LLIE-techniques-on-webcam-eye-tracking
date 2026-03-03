import os


FACEGAZE_ROOT = "data/original/MPIIFaceGaze"


def load_facegaze_samples():

    samples = []

    for subject in sorted(os.listdir(FACEGAZE_ROOT)):

        subject_path = os.path.join(FACEGAZE_ROOT, subject)

        if not os.path.isdir(subject_path):
            continue

        annotation_file = os.path.join(subject_path, f"{subject}.txt")

        if not os.path.exists(annotation_file):
            continue

        with open(annotation_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 27:
                    continue

                image_rel_path = parts[0]  # e.g., day01/0001.jpg

                full_rel_path = os.path.normpath(
                    os.path.join(subject, image_rel_path)
                )

                samples.append({
                    "subject": subject,
                    "path": full_rel_path,
                    "annotation": parts
                })

    return samples