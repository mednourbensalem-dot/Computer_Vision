"""
People Counting
IP Webcam (phone) + MobileNet-SSD + centroid tracker + 7-segment UI

Usage:
    python people_counting.py --source http://192.168.1.20:8080/video
    python people_counting.py --source 0            # laptop webcam
Keys: q = quit | r = reset counters
"""
import argparse
import os
from collections import OrderedDict

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.join(HERE, "models", "deploy.prototxt")
MODEL = os.path.join(HERE, "models", "mobilenet_iter_73000.caffemodel")
PERSON_ID = 15  # "person" class index in MobileNet-SSD (VOC)

# ---------- Layout ----------
W, H = 640, 480   # video size
TITLE_H = 70      # green title bar
PANEL_H = 150     # OUT / IN panel
STATUS_H = 32     # status bar
TOTAL_H = TITLE_H + H + PANEL_H + STATUS_H


# ---------- Tracker ----------
class CentroidTracker:
    def __init__(self, max_disappeared=30, max_dist=90):
        self.next_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_dist = max_dist

    def register(self, c):
        self.objects[self.next_id] = c
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, oid):
        del self.objects[oid]
        del self.disappeared[oid]

    def update(self, centroids):
        if len(centroids) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        inputs = np.array(centroids)

        if len(self.objects) == 0:
            for c in inputs:
                self.register(tuple(c))
            return self.objects

        ids = list(self.objects.keys())
        cur = np.array(list(self.objects.values()))
        D = np.linalg.norm(cur[:, np.newaxis] - inputs[np.newaxis, :], axis=2)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        used_r, used_c = set(), set()

        for r, c in zip(rows, cols):
            if r in used_r or c in used_c:
                continue
            if D[r, c] > self.max_dist:  # too far: not the same person
                continue
            oid = ids[r]
            self.objects[oid] = tuple(inputs[c])
            self.disappeared[oid] = 0
            used_r.add(r)
            used_c.add(c)

        for r in set(range(D.shape[0])) - used_r:
            oid = ids[r]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.deregister(oid)

        for c in set(range(D.shape[1])) - used_c:
            self.register(tuple(inputs[c]))

        return self.objects


# ---------- Detector ----------
class PersonDetector:
    def __init__(self, conf_min):
        self.conf_min = conf_min
        self.use_ssd = True
        try:
            self.net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)
        except Exception:
            self.use_ssd = False
            print("[WARN] MobileNet-SSD model not found -> falling back to HOG "
                  "(less accurate). Run: python download_models.py")
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame):
        """Return list of boxes (x1, y1, x2, y2)."""
        boxes = []
        if self.use_ssd:
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                         0.007843, (300, 300), 127.5)
            self.net.setInput(blob)
            det = self.net.forward()
            for i in range(det.shape[2]):
                conf = det[0, 0, i, 2]
                if conf > self.conf_min and int(det[0, 0, i, 1]) == PERSON_ID:
                    b = det[0, 0, i, 3:7] * np.array([W, H, W, H])
                    x1, y1, x2, y2 = b.astype("int")
                    boxes.append((x1, y1, x2, y2))
        else:
            rects, _ = self.hog.detectMultiScale(
                frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
            for (x, y, w, h) in rects:
                boxes.append((x, y, x + w, y + h))
        return boxes


# ---------- 7-segment drawing ----------
SEGS = {
    "0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg",
    "5": "acdfg", "6": "acdefg", "7": "abc", "8": "abcdefg", "9": "abcdfg",
}


def draw_digit(img, ch, x, y, w, h, on=(20, 20, 20), off=(215, 215, 215)):
    t = max(6, w // 5)
    rects = {
        "a": (x + t, y, x + w - t, y + t),
        "b": (x + w - t, y + t, x + w, y + h // 2),
        "c": (x + w - t, y + h // 2, x + w, y + h - t),
        "d": (x + t, y + h - t, x + w - t, y + h),
        "e": (x, y + h // 2, x + t, y + h - t),
        "f": (x, y + t, x + t, y + h // 2),
        "g": (x + t, y + h // 2 - t // 2, x + w - t, y + h // 2 + t // 2),
    }
    active = SEGS.get(ch, "")
    for name, (x1, y1, x2, y2) in rects.items():
        cv2.rectangle(img, (x1, y1), (x2, y2), on if name in active else off, -1)


def draw_counter(img, x, y, w, h, label, value, bg):
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    cv2.putText(img, label, (x + 15, y + h // 2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
    px, py, pw, ph = x + w - 175, y + 20, 160, h - 40
    cv2.rectangle(img, (px, py), (px + pw, py + ph), (215, 215, 215), -1)
    text = f"{min(value, 999):03d}"
    for i, ch in enumerate(text):
        draw_digit(img, ch, px + 10 + i * 50, py + 12, 38, ph - 24)


def build_screen(frame, count_in, count_out):
    screen = np.zeros((TOTAL_H, W, 3), np.uint8)

    # title
    cv2.rectangle(screen, (0, 0), (W, TITLE_H), (0, 110, 0), -1)
    (tw, th), _ = cv2.getTextSize("PEOPLE COUNTING", cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)
    cv2.putText(screen, "PEOPLE COUNTING", ((W - tw) // 2, (TITLE_H + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)

    # video
    screen[TITLE_H:TITLE_H + H] = frame

    # OUT / IN panel
    py = TITLE_H + H
    cv2.rectangle(screen, (0, py), (W, py + PANEL_H), (190, 190, 190), -1)
    draw_counter(screen, 10, py + 10, 305, PANEL_H - 20, "OUT", count_out, (0, 0, 139))
    draw_counter(screen, 325, py + 10, 305, PANEL_H - 20, "IN", count_in, (0, 110, 0))

    # status bar
    sy = py + PANEL_H
    cv2.rectangle(screen, (0, sy), (W, TOTAL_H), (240, 240, 240), -1)
    cv2.putText(screen, "Status: Running   (q = quit | r = reset)", (10, sy + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return screen


# ---------- Main ----------
def parse_args():
    p = argparse.ArgumentParser(description="People counting with MobileNet-SSD")
    p.add_argument("--source", default="http://10.142.13.227:8080/video",
                   help="IP Webcam URL, video file path, or camera index (0, 1...)")
    p.add_argument("--conf", type=float, default=0.5, help="detection confidence")
    p.add_argument("--invert", action="store_true",
                   help="swap IN/OUT direction")
    return p.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    camera = cv2.VideoCapture(source)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        print(f"[ERROR] Cannot open source: {args.source}\n"
              "Check: phone & PC on the same Wi-Fi, IP Webcam server started, "
              "correct IP in --source (ends with /video).")
        raise SystemExit(1)

    detector = PersonDetector(args.conf)
    tracker = CentroidTracker()
    last_side = {}
    count_in = count_out = 0
    line_x = W // 2

    win = "People Counting"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, TOTAL_H)

    while True:
        ret, frame = camera.read()
        if not ret:
            print("[ERROR] Camera read failed / stream ended")
            break

        frame = cv2.resize(frame, (W, H))
        boxes = detector.detect(frame)

        centroids = []
        for (x1, y1, x2, y2) in boxes:
            centroids.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        objects = tracker.update(centroids)

        for oid, (cx, cy) in objects.items():
            side = cx > line_x
            if oid in last_side:
                crossed_right = side and not last_side[oid]
                crossed_left = (not side) and last_side[oid]
                if args.invert:
                    crossed_right, crossed_left = crossed_left, crossed_right
                if crossed_right:
                    count_in += 1
                elif crossed_left:
                    count_out += 1
            last_side[oid] = side
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"ID {oid}", (cx - 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # forget IDs that disappeared
        for oid in list(last_side.keys()):
            if oid not in objects:
                del last_side[oid]

        # counting line
        cv2.line(frame, (line_x, 0), (line_x, H), (255, 0, 0), 3)

        cv2.imshow(win, build_screen(frame, count_in, count_out))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            count_in = count_out = 0

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
