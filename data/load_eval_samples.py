import os

EVAL_ROOT = os.path.normpath(
    "data/original/MPIIGaze/Evaluation Subset/sample list for eye image"
)

def load_evaluation_samples():
    samples = []
    for subject_file in sorted(os.listdir(EVAL_ROOT)):
        if not subject_file.endswith(".txt"):
            continue

        subject_id = subject_file.replace(".txt", "")
        txt_path = os.path.join(EVAL_ROOT, subject_file)

        with open(txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                image_rel = parts[0]  # dayXX/XXXX.jpg
                eye = parts[1]        # left / right

                full_rel_path = os.path.normpath(
                    os.path.join(subject_id, image_rel)
                )

                samples.append({
                    "path": full_rel_path,
                    "eye": eye
                })
    return samples
