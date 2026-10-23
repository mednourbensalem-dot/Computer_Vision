# People Counting

Real-time people counter using a phone camera (IP Webcam app), MobileNet-SSD
detection, a centroid tracker, and a 7-segment style IN / OUT display.

A person crossing the blue vertical line left → right counts as **IN**,
right → left counts as **OUT** (use `--invert` to swap).

## Project structure

```
people-counting/
├── people_counting.py    # main app
├── download_models.py    # fetches the model files into models/
├── models/
│   ├── deploy.prototxt
│   └── mobilenet_iter_73000.caffemodel
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python download_models.py      # only if models/ is empty
```

## Run

1. Install **IP Webcam** on your Android phone, open it, tap **Start server**.
2. Phone and PC must be on the **same Wi-Fi**. Note the IP shown (e.g. `192.168.1.20:8080`).
3. Run:

```bash
python people_counting.py --source http://192.168.1.20:8080/video
```

Other sources:

```bash
python people_counting.py --source 0              # laptop webcam
python people_counting.py --source video.mp4      # video file
python people_counting.py --source 0 --invert     # swap IN/OUT
python people_counting.py --source 0 --conf 0.6   # stricter detection
```

Keys: `q` quit, `r` reset counters.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot open source` | Check the IP, same Wi-Fi, URL ends with `/video`; test it in a browser |
| Falls back to HOG | Model files missing → `python download_models.py` |
| Lag / delay | Lower resolution in IP Webcam settings |
| Double counts | Raise `--conf`, or increase `max_disappeared` in `CentroidTracker` |

## License
MIT
