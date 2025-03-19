import gdown
import urllib.request

# Dropbox direct download links (modify '?dl=0' to '?dl=1')
files = {
    "models/yolov8x.pt": "https://www.dropbox.com/scl/fi/b65zgfg9ew9owtyp4n4hg/yolov8x.pt?rlkey=7perigcdbboe62b1u891t14x9&dl=1",
    "models/facenet512_weights.h5": "https://www.dropbox.com/scl/fi/c9haq38tmme4ujbstarsu/facenet512_weights.h5?rlkey=bnig5qfv16nimdrw2xk8ifoc4&dl=1",
    "models/facenet_weights.h5": "https://www.dropbox.com/scl/fi/39vgh234jo6ug8f6athjb/facenet_weights.h5?rlkey=ok9zsdhypezox1fceuv81rxvx&dl=1"
}

for file_path, file_url in files.items():
    print(f"Downloading {file_path}...")
    urllib.request.urlretrieve(file_url, file_path)
    print(f"Downloaded {file_path}")

print("All model files downloaded successfully!")
