import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(path_files, root_dir):
    if path_files.find('.txt') != -1:
        paths, size = make_dataset_txt(path_files, root_dir)
    else:
        paths, size = make_dataset_dir(path_files)

    return paths, size

def make_dataset_txt(path_files, root_dir):
    # reading txt file
    image_paths = []

    with open(path_files) as f:
        paths = f.readlines()

    for path in paths:
        # path = path.strip()
        path = root_dir + path.strip()
        image_paths.append(path)

    min_n = 8000
    if False:
        if len(image_paths) > min_n:
            image_paths = image_paths[:min_n]

    return image_paths, len(image_paths)


def make_dataset_dir(dir):
    image_paths = []

    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in os.walk(dir):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                image_paths.append(path)

    min_n = 8000
    if False:
        if len(image_paths) > min_n:
            image_paths = image_paths[:min_n]

    return image_paths, len(image_paths)
