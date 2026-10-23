"""Download the MobileNet-SSD files into ./models (skips files already present)."""
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {
    "deploy.prototxt":
        "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/voc/MobileNetSSD_deploy.prototxt",
    "mobilenet_iter_73000.caffemodel":
        "https://raw.githubusercontent.com/PINTO0309/MobileNet-SSD-RealSense/master/caffemodel/MobileNetSSD/MobileNetSSD_deploy.caffemodel",
}

os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
for name, url in FILES.items():
    path = os.path.join(HERE, "models", name)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        print(f"[skip] {name} already exists")
        continue
    print(f"[download] {name} ...")
    urllib.request.urlretrieve(url, path)
    print(f"[ok] {path} ({os.path.getsize(path) // 1024} KB)")
print("Done.")
