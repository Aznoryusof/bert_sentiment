import urllib.request
import os, sys
import zipfile


from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()


MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(MAIN_DIR, "data/")
sys.path.append(MAIN_DIR)


def unzip_file(path):
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(DATA_DIR))


def dl_hotel_reviews():
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)

    files = [
        (
            "data/",
            "jiashenliu/515k-hotel-reviews-data-in-europe"
        )
    ]

    for dir, dataset in files:
        if not os.path.exists(os.path.join(DATA_DIR, "515k-hotel-reviews-data-in-europe.zip")):
            print("Downloading to", dir)

            api.dataset_download_files(dataset, dir)
            unzip_file(os.path.join(DATA_DIR, "515k-hotel-reviews-data-in-europe.zip"))

            print('DONE.')

        else:
            print("Data already available.")


if __name__ == "__main__":
    dl_hotel_reviews()


    