import os, sys

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(MAIN_DIR, "data/")
RESULT_DIR = os.path.join(MAIN_DIR, "result/")
sys.path.append(MAIN_DIR)

import urllib.request
from google_drive_downloader import GoogleDriveDownloader as gdd


def _download_raw_dataset():
    cwd = os.getcwd()
    gdd.download_file_from_google_drive(file_id='1jnvfJSOJpZ2Itq-44zGjrzL4mRiydvPx',
                                        dest_path=os.path.join(cwd, "result.zip"),
                                        unzip=True)

    return print("Model artifacts downloaded: {}".format(cwd))


if __name__ == '__main__':
    _download_raw_dataset()