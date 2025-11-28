import kagglehub

# Download latest version
def download_dataset(download_path="zq1200/magnus-carlsen-complete-chess-games-20012022"):
    path = kagglehub.dataset_download(download_path)
    print("Path to dataset files:", path)

download_dataset()