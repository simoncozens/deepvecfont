# data loader for training main model
import os
import pickle
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T

from data_utils.svg_utils import MAX_PATH_COMMANDS

torch.multiprocessing.set_sharing_strategy("file_system")


class SVGDataset(data.Dataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.mode = kwargs.get("mode", "train")
        self.img_size = kwargs.get("img_size", 128)
        self.char_num = kwargs.get("char_num", 52)
        self.max_seq_len = MAX_PATH_COMMANDS + 1
        self.feature_dim = kwargs.get("seq_feature_dim", 10)
        self.trans = kwargs.get("transform")


class SVGDatasetDirs(SVGDataset):
    def __init__(self, root_path, **kwargs):
        super().__init__(
            **kwargs,
        )
        self.font_paths = []
        self.dir_path = os.path.join(root_path, self.mode)
        for root, dirs, files in os.walk(self.dir_path):
            for dir_name in dirs:
                self.font_paths.append(os.path.join(self.dir_path, dir_name))
        self.font_paths.sort()
        print(f"Finished loading {kwargs["mode"]} paths")

    def __getitem__(self, index):
        font_path = self.font_paths[index]
        rendered = (
            torch.FloatTensor(
                np.load(
                    os.path.join(font_path, "rendered_" + str(self.img_size) + ".npy")
                )
            ).view(self.char_num, self.img_size, self.img_size)
            / 255.0
        )

        return {
            "class": torch.LongTensor(np.load(os.path.join(font_path, "class.npy"))),
            "seq_len": torch.LongTensor(
                np.load(os.path.join(font_path, "seq_len.npy"))
            ),
            "sequence": torch.FloatTensor(
                np.load(os.path.join(font_path, "sequence.npy"))
            ).view(self.char_num, self.max_seq_len, self.feature_dim),
            "rendered": self.trans(rendered),
            "font_id": torch.FloatTensor(
                np.load(os.path.join(font_path, "font_id.npy")).astype(np.float32)
            ),
        }

    def __len__(self):
        return len(self.font_paths)


class SVGDatasetPickle(SVGDataset):
    def __init__(
        self,
        root_path,
        mode="train",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pkl_path = os.path.join(root_path, self.mode, f"{mode}_all.pkl")
        pkl_f = open(self.pkl_path, "rb")
        print(f"Loading {self.pkl_path} pickle file ...")
        self.all_fonts = pickle.load(pkl_f)
        pkl_f.close()
        print("Finished loading pkls")

    def __getitem__(self, index):
        cur_glyph = self.all_fonts[index]
        return {
            "class": torch.LongTensor(cur_glyph["class"]),
            "seq_len": torch.LongTensor(cur_glyph["seq_len"]),
            "sequence": torch.FloatTensor(cur_glyph["sequence"]).view(
                self.char_num, self.max_seq_len, self.feature_dim
            ),
            "rendered": self.trans(
                torch.FloatTensor(cur_glyph["rendered"]).view(
                    self.char_num, self.img_size, self.img_size
                )
                / 255.0
            ),
            "font_id": torch.FloatTensor([float(cur_glyph["binary_fp"])]),
        }

    def __len__(self):
        return len(self.all_fonts)


def get_loader(
    root_path,
    img_size,
    char_num,
    seq_feature_dim,
    batch_size,
    read_mode,
    mode="train",
):
    # SetRange = T.Lambda(lambda X: 2 * X - 1.)  # convert [0, 1] -> [-1, 1]
    SetRange = T.Lambda(lambda X: 1.0 - X)  # convert [0, 1] -> [0, 1]
    transform = T.Compose([SetRange])
    if read_mode == "dirs":
        dataset = SVGDatasetDirs(
            root_path,
            img_size=img_size,
            char_num=char_num,
            seq_feature_dim=seq_feature_dim,
            transform=transform,
            mode=mode,
        )
    else:
        dataset = SVGDatasetPickle(
            root_path,
            img_size=img_size,
            char_num=char_num,
            seq_feature_dim=seq_feature_dim,
            transform=transform,
            mode=mode,
        )
    dataloader = data.DataLoader(
        dataset,
        batch_size,
        shuffle=(mode == "train"),
        num_workers=0,
        drop_last=True,
    )
    return dataloader


if __name__ == "__main__":
    ROOT_PATH = "data/vecfont_dataset_dirs"
    SEQ_FEATURE_DIM = 10
    BATCH_SIZE = 1
    CHAR_NUM = 52
    IMG_SIZE = 128

    loader = get_loader(
        ROOT_PATH, IMG_SIZE, CHAR_NUM, SEQ_FEATURE_DIM, BATCH_SIZE, "dirs", "train"
    )
    fout = open("train_id_record_old.txt", "w")
    for idx, batch in enumerate(loader):
        import IPython

        IPython.embed()
        binary_fp = batch["font_id"].numpy()[0][0]
        fout.write("%05d" % int(binary_fp) + "\n")
