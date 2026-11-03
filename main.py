from pathlib import Path


def main():
    data_dir = Path(__file__).resolve().parent / "data"
    #for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
    #    image_count = sum(1 for path in class_dir.iterdir() if path.is_file())
    #    print(f"{class_dir.name}: {image_count} images")


if __name__ == "__main__":
    main()