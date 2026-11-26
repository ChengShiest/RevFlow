import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from revflow.data import ImageNet

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("imagenet_prep", ROOT / "data_prep" / "imagenet.py")
imagenet_prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imagenet_prep)


class DataPrepTest(unittest.TestCase):
    def test_remap_matches_imagefolder_order(self):
        labels = np.array([2, 0, 1, 0, 2, 2, 1, 0])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for row, label in enumerate(labels):
                folder = out / "train" / f"{label:04d}"
                folder.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), (row, 0, 0)).save(folder / f"{row:07d}.JPEG")
            np.savez(out / "train_rows.npz", labels=labels, lengths=np.ones_like(labels))
            rows = np.array([5, 0, 3, 6])
            np.save(out / "rows.npy", rows)
            imagenet_prep.remap(SimpleNamespace(out=out, rows=out / "rows.npy", range=None, output=out / "ix.npy"))
            indices = np.load(out / "ix.npy")
            dataset = ImageNet(out / "train")
            for row, index in zip(rows, indices):
                path, label = dataset.samples[index]
                self.assertEqual(path.name, f"{row:07d}.JPEG")
                self.assertEqual(label, labels[row])

    def test_support_rows_shipped(self):
        rows = np.load(ROOT / "data_prep" / "nm_support" / "rows.npy")
        labels = np.load(ROOT / "data_prep" / "nm_support" / "labels.npy")
        self.assertEqual(rows.shape, (51200,))
        self.assertEqual(len(np.unique(rows)), 51200)
        self.assertEqual(np.bincount(labels).min(), 51)
        self.assertEqual(imagenet_prep.digest(rows), imagenet_prep.EXPECTED["nm_support_rows_sha256"])


if __name__ == "__main__":
    unittest.main()
