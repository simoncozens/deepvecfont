import argparse
import os
from pathlib import Path

import numpy as np
import svg_utils
from PIL import Image, ImageDraw, ImageFont
from extract_path import extract_path, make_hb_font
import tqdm


def create_db(opts):
    print("Process ttf to npy files in dirs....")
    basedir = Path(opts.ttf_path) / opts.split
    all_font_paths = list(basedir.glob("*.ttf"))
    num_fonts = len(all_font_paths)
    num_fonts_w = len(str(num_fonts))
    print(f"Number {opts.split} fonts before processing", num_fonts)

    for i, font_path in tqdm.tqdm(enumerate(all_font_paths), total=num_fonts):
        cur_font_glyphs = load_font_glyphs(opts.charset, font_path, opts.img_size)

        if cur_font_glyphs is None:
            print("skipping font", font_path)
            continue

        try:
            font = ImageFont.truetype(font_path, opts.img_size, encoding="unic")
        except Exception:
            print("cant open " + str(font_path))
            continue
        # use the font whose all glyphs are valid
        # merge the whole font
        sequence = []
        seq_len = []
        binaryfp = []
        char_class = []
        rendered = []
        ok = True
        for charid, char in enumerate(opts.charset):
            example = cur_font_glyphs[charid]
            sequence.append(example["sequence"])
            seq_len.append(example["seq_len"])
            char_class.append(example["class"])
            rendering = render_glyph(font, char, opts.img_size)
            if rendering is None:
                print("skipping glyph", char)
                ok = False
                break
            binaryfp = example["binary_fp"]
            rendered.append(rendering)
        if not ok:
            print("skipping font (rendering failure)", font_path)
            continue
        rendered = np.array(rendered)
        output_dir = (
            Path(opts.output_path)
            / opts.split
            / "{num:0{width}}".format(num=i, width=num_fonts_w)
        )
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        np.save(output_dir / "sequence.npy", np.array(sequence))
        np.save(output_dir / "seq_len.npy", np.array(seq_len))
        np.save(output_dir / "class.npy", np.array(char_class))
        np.save(output_dir / "font_id.npy", np.array(binaryfp))
        np.save(output_dir / f"rendered_{opts.img_size}.npy", rendered)

    print(
        "Finished processing all sfd files, logs (invalid glyphs and paths) are saved to",
        opts.log_dir,
    )


def render_glyph(font, char, img_size):
    """Render a single glyph to an image."""
    # Create a blank image with white background
    img = Image.new("L", (img_size, img_size), 255)
    draw = ImageDraw.Draw(img)
    try:
        l, t, r, b = font.getbbox(char)
    except Exception:
        print("cannot get bbox")
        return
    vbox_w = r - l
    vbox_h = b - t
    aspect_ratio = float(img_size) / max(int(vbox_w), int(vbox_h))

    if int(vbox_h) > int(vbox_w):
        add_to_y = 0
        add_to_x = abs(int(vbox_h) - int(vbox_w)) / 2
        add_to_x = add_to_x * aspect_ratio
    else:
        add_to_y = abs(int(vbox_h) - int(vbox_w)) / 2
        add_to_y = add_to_y * aspect_ratio
        add_to_x = 0

    array = np.ndarray((img_size, img_size), np.uint8)
    array[:, :] = 255
    image = Image.fromarray(array)
    draw = ImageDraw.Draw(image)

    try:
        ascent, _ = font.getmetrics()
    except Exception:
        print("cannot get ascent, descent")
        return

    draw.text(
        (
            add_to_x,
            add_to_y + img_size - ascent - int((img_size / 24.0) * (4.0 / 3.0)),
        ),
        char,
        (0),
        font=font,
    )
    # Convert the image to a numpy array
    return np.array(image)


def load_font_glyphs(charset, font_path, img_size):
    font, upem = make_hb_font(font_path)
    good_paths = []

    for char in charset:
        # Extract char as SVG string
        svg = extract_path(font, char, upem)
        pathunibfp = svg, ord(char), font_path
        if not svg_utils.is_valid_path(pathunibfp):
            return None
        good_paths.append(pathunibfp)

    # Now we know the whole font is valid, we can process all glyphs
    return [svg_utils.create_example(pathunibfp, img_size) for pathunibfp in good_paths]


def cal_mean_stddev(opts):
    print("Calculating all glyphs' mean stddev ....")
    font_paths = []
    dir_path = os.path.join(opts.output_path, opts.split)
    for _root, dirs, _files in os.walk(dir_path):
        for dir_name in dirs:
            font_paths.append(os.path.join(dir_path, dir_name))
    font_paths.sort()
    num_fonts = len(font_paths)
    num_chars = len(opts.charset)
    main_stddev_accum = svg_utils.MeanStddev()
    cur_sum_count = main_stddev_accum.create_accumulator()
    for i in tqdm.tqdm(range(0, num_fonts)):
        cur_font_path = font_paths[i]
        seqlens = np.load(os.path.join(cur_font_path, "seq_len.npy")).tolist()
        sequences = np.load(os.path.join(cur_font_path, "sequence.npy")).tolist()
        for charid in range(num_chars):
            cur_font_char = {}
            cur_font_char["seq_len"] = seqlens[charid]
            cur_font_char["sequence"] = sequences[charid]
            cur_sum_count = main_stddev_accum.add_input(cur_sum_count, cur_font_char)

    output = main_stddev_accum.extract_output(cur_sum_count)
    mean = output["mean"]
    stdev = output["stddev"]
    mean = np.concatenate((np.zeros([4]), mean[4:]), axis=0)
    stdev = np.concatenate((np.ones([4]), stdev[4:]), axis=0)
    # finally, save the mean and stddev files
    np.save(os.path.join(opts.output_path, opts.split, "mean"), mean)
    np.save(os.path.join(opts.output_path, opts.split, "stdev"), stdev)

    # rename npy to npz, to match the file names predefined in main.py
    path_ms = os.path.join(opts.output_path, opts.split)
    os.rename(os.path.join(path_ms, "mean.npy"), os.path.join(path_ms, "mean.npz"))
    os.rename(os.path.join(path_ms, "stdev.npy"), os.path.join(path_ms, "stdev.npz"))


def main():
    parser = argparse.ArgumentParser(description="LMDB creation")
    parser.add_argument(
        "--charset",
        type=str,
        default="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    )
    parser.add_argument("--ttf_path", type=str, default="font_ttfs")
    parser.add_argument(
        "--output_path",
        type=str,
        default="../data/vecfont_dataset_dirs_/",
        help="Path to write the database to",
    )
    parser.add_argument(
        "--img_size", type=int, default=128, help="the height and width of glyph images"
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--log_dir", type=str, default="./font_sfds/log/")
    parser.add_argument(
        "--phase",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="0 all, 1 create db, 2 cal stddev",
    )

    opts = parser.parse_args()
    split_path = os.path.join(opts.output_path, opts.split)

    if not os.path.exists(split_path):
        os.makedirs(split_path)

    if not os.path.exists(opts.log_dir):
        os.makedirs(opts.log_dir)

    if opts.phase <= 1:
        create_db(opts)

    if opts.phase <= 2 and opts.split == "train":
        cal_mean_stddev(opts)


if __name__ == "__main__":
    main()
