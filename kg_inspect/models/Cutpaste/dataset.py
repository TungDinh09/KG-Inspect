from paddle.io import Dataset
from pathlib import Path
from PIL import Image
from joblib import Parallel, delayed


class Repeat(Dataset):
    def __init__(self, org_dataset, new_length):
        super().__init__()
        self.org_dataset = org_dataset
        self.org_length = len(self.org_dataset)
        self.new_length = new_length

    def __len__(self):
        return self.new_length

    def __getitem__(self, idx):
        return self.org_dataset[idx % self.org_length]


class MVTecAT(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, defect_name, size, transform=None, mode="train"):
        """
        Args:
            root_dir (string): Directory with the MVTec AD dataset.
            defect_name (string): defect to load.
            transform: Transform to apply to data
            mode: "train" loads training samples "test" test samples default "train"
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.defect_name = defect_name
        self.transform = transform
        self.mode = mode
        self.size = size

        # find test images
        if self.mode == "train":
            self.image_names = list((self.root_dir / defect_name / "train" / "good").glob("*.png"))
            print("loading images")
            # during training we cache the smaller images for performance reasons (not a good coding style)
            # self.imgs = [Image.open(file).resize((size,size)).convert("RGB") for file in self.image_names]
            self.imgs = Parallel(n_jobs=10)(
                delayed(lambda file: Image.open(file).resize((size, size)).convert("RGB"))(file) for file in self.image_names)
            print(f"loaded {len(self.imgs)} images")
        else:
            # test mode
            self.image_names = list((self.root_dir / defect_name / "test").glob(str(Path("*") / "*.png")))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.mode == "train":
            # img = Image.open(self.image_names[idx])
            # img = img.convert("RGB")
            img = self.imgs[idx].copy()
            if self.transform is not None:
                img = self.transform(img)
            return img
        else:
            filename = self.image_names[idx]
            label = filename.parts[-2]
            img = Image.open(filename)
            img = img.resize((self.size, self.size)).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            return img, label != "good"
class VisA(Dataset):
    """VisA Dataset compatible with CutPaste training"""

    def __init__(self, root_dir, defect_name, size, transform=None, mode="train", label_file=None):
        """
        Args:
            root_dir (string): Path to VisA dataset root (e.g. "path/to/VisA")
            defect_name (string): subfolder (e.g. "capsules")
            size (int): image resize size
            transform: Transform to apply
            mode (string): "train" or "test"
            label_file (string, optional): path to label txt file if needed
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.defect_name = defect_name
        self.transform = transform
        self.mode = mode
        self.size = size

        base_path = self.root_dir / defect_name / "Data" / "Images"

        # TRAIN MODE: only Normal images
        if self.mode == "train":
            self.image_names = list((base_path / "Normal").glob("*.JPG"))
            print(f"Loading {len(self.image_names)} normal training images...")
            self.imgs = Parallel(n_jobs=10)(
                delayed(lambda f: Image.open(f).resize((size, size)).convert("RGB"))(file)
                for file in self.image_names
            )
        else:
            # TEST MODE: Normal + Anomaly images
            self.image_names = list((base_path / "Normal").glob("*.JPG")) + \
                               list((base_path / "Anomaly").glob("*.JPG"))
            self.labels = []
            for f in self.image_names:
                if "Normal" in str(f):
                    self.labels.append("normal")
                else:
                    # if you have label file, could load from there
                    self.labels.append("anomaly")

        # Optional: parse external label file
        if label_file and Path(label_file).exists():
            self._load_labels_from_file(label_file)

    def _load_labels_from_file(self, file_path):
        label_map = {}
        with open(file_path, "r") as f:
            for line in f:
                path, label = line.strip().split("\t")
                label_map[Path(path).name] = label
        # override test labels
        if self.mode != "train":
            self.labels = [label_map.get(p.name, "anomaly") for p in self.image_names]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.mode == "train":
            img = self.imgs[idx].copy()
            if self.transform:
                img = self.transform(img)
            return img
        else:
            img_path = self.image_names[idx]
            label = self.labels[idx]
            img = Image.open(img_path).resize((self.size, self.size)).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label != "normal"   # Boolean: True = anomaly


class VisA(Dataset):
    """VisA Dataset compatible with CutPaste training"""

    def __init__(self, root_dir, defect_name, size, transform=None, mode="train", label_file=None):
        """
        Args:
            root_dir (string): Path to VisA dataset root (e.g. "path/to/VisA")
            defect_name (string): subfolder (e.g. "capsules")
            size (int): image resize size
            transform: Transform to apply
            mode (string): "train" or "test"
            label_file (string, optional): path to label txt file if needed
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.defect_name = defect_name
        self.transform = transform
        self.mode = mode
        self.size = size

        base_path = self.root_dir / defect_name / "Data" / "Images"

        # TRAIN MODE: only Normal images
        if self.mode == "train":
            self.image_names = list((base_path / "Normal").glob("*.JPG"))
            print(f"Loading {len(self.image_names)} normal training images...")
            self.imgs = Parallel(n_jobs=10)(
                delayed(lambda f: Image.open(f).resize((size, size)).convert("RGB"))(file)
                for file in self.image_names
            )
        else:
            # TEST MODE: Normal + Anomaly images
            self.image_names = list((base_path / "Normal").glob("*.JPG")) + \
                               list((base_path / "Anomaly").glob("*.JPG"))
            self.labels = []
            for f in self.image_names:
                if "Normal" in str(f):
                    self.labels.append("normal")
                else:
                    # if you have label file, could load from there
                    self.labels.append("anomaly")

        # Optional: parse external label file
        if label_file and Path(label_file).exists():
            self._load_labels_from_file(label_file)

    def _load_labels_from_file(self, file_path):
        label_map = {}
        with open(file_path, "r") as f:
            for line in f:
                path, label = line.strip().split("\t")
                label_map[Path(path).name] = label
        # override test labels
        if self.mode != "train":
            self.labels = [label_map.get(p.name, "anomaly") for p in self.image_names]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.mode == "train":
            img = self.imgs[idx].copy()
            if self.transform:
                img = self.transform(img)
            return img
        else:
            img_path = self.image_names[idx]
            label = self.labels[idx]
            img = Image.open(img_path).resize((self.size, self.size)).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label != "normal"   # Boolean: True = anomaly
