import pathlib
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import ballot  # noqa: E402

ASSET = pathlib.Path(__file__).parent / "assets" / "benchmark_infographic.jpg"
PEN = (170, 40, 30)  # blue ballpoint, BGR


def test_benchmark_infographic():
    res = ballot.analyze(cv2.imread(str(ASSET)))
    boxes = res["boxes"]
    assert len(boxes) == 32
    mid = np.median([b["bbox"][0] for b in boxes])
    left = [b for b in boxes if b["bbox"][0] < mid]
    right = [b for b in boxes if b["bbox"][0] >= mid]
    assert len(left) == len(right) == 16
    assert all(b["label"] == "VALID" for b in left), [(b["number"], b["reason"]) for b in left if b["label"] != "VALID"]
    top_right = min(right, key=lambda b: b["bbox"][1] - b["bbox"][0])  # top row, rightmost column
    assert top_right["label"] == "EMPTY"
    others = [b for b in right if b is not top_right]
    assert all(b["label"] == "INVALID" for b in others), [b["number"] for b in others if b["label"] != "INVALID"]


# ---- synthetic ballots mimicking the EC 2569 layouts ----

BW, BH = 100, 50


def cross(img, box, offset=(0, 0)):
    x, y, w, h = box
    cx, cy = x + w // 2 + offset[0], y + h // 2 + offset[1]
    cv2.line(img, (cx - 20, cy - 15), (cx + 20, cy + 15), PEN, 3)
    cv2.line(img, (cx - 20, cy + 15), (cx + 20, cy - 15), PEN, 3)


def green_ballot(rows=5, cols=4):
    """ส.ส. แบบแบ่งเขต style: [filled number cell | mark box] x 4 columns, no-vote box below."""
    img = np.full((rows * 70 + 330, 1000, 3), 255, np.uint8)
    cv2.putText(img, "BALLOT", (380, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    boxes = []
    for c in range(cols):
        for r in range(rows):
            x, y = 60 + c * 230, 120 + r * 70
            cv2.rectangle(img, (x, y), (x + BW, y + BH), (60, 140, 40), -1)
            cv2.putText(img, str(c * rows + r + 1), (x + 30, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            box = (x + BW, y, BW, BH)
            cv2.rectangle(img, box[:2], (box[0] + BW, y + BH), (0, 0, 0), 3)
            boxes.append(box)
    y = 120 + rows * 70 + 80
    cv2.putText(img, "NO VOTE", (560, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    nv = (760, y, BW, BH)
    cv2.rectangle(img, nv[:2], (nv[0] + BW, y + BH), (0, 0, 0), 3)
    return img, boxes, nv


def pink_ballot(rows=5):
    """ส.ส. แบบบัญชีรายชื่อ style: [number | logo | party name | mark box], flush cells, 2 columns."""
    img = np.full((rows * 70 + 250, 1000, 3), (225, 205, 250), np.uint8)
    boxes = []
    for c in range(2):
        for r in range(rows):
            x, y = 40 + c * 480, 100 + r * 70
            cv2.rectangle(img, (x, y), (x + 70, y + BH), (0, 0, 0), -1)
            cv2.putText(img, str(c * rows + r + 1), (x + 15, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            for cx, cw in ((x + 70, 60), (x + 130, 170)):
                cv2.rectangle(img, (cx, y), (cx + cw, y + BH), (255, 255, 255), -1)
                cv2.rectangle(img, (cx, y), (cx + cw, y + BH), (0, 0, 0), 2)
            cv2.circle(img, (x + 100, y + 25), 15, (0, 0, 180), -1)
            cv2.putText(img, f"PARTY {c * rows + r + 1}", (x + 140, y + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            box = (x + 300, y, BW, BH)
            cv2.rectangle(img, box[:2], (box[0] + BW, y + BH), (255, 255, 255), -1)
            cv2.rectangle(img, box[:2], (box[0] + BW, y + BH), (0, 0, 0), 2)
            boxes.append(box)
    y = 100 + rows * 70 + 60
    nv = (820, y, BW, BH)
    cv2.rectangle(img, nv[:2], (nv[0] + BW, y + BH), (255, 255, 255), -1)
    cv2.rectangle(img, nv[:2], (nv[0] + BW, y + BH), (0, 0, 0), 2)
    return img, boxes, nv


def rotate(img, deg):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1)
    return cv2.warpAffine(img, m, (w, h), borderValue=(255, 255, 255))


@pytest.mark.parametrize("make", [green_ballot, pink_ballot])
def test_synthetic_layouts(make):
    img, boxes, nv = make()
    blank = ballot.analyze(img)
    assert blank["counts"]["total"] == len(boxes) + 1, "label/logo cells must not count as mark boxes"
    assert sum(b["is_no_vote"] for b in blank["boxes"]) == 1
    assert blank["verdict"] == ballot.VERDICT_NOVOTE

    voted = img.copy()
    cross(voted, boxes[6])
    jpeg = cv2.imdecode(cv2.imencode(".jpg", voted, [cv2.IMWRITE_JPEG_QUALITY, 40])[1], cv2.IMREAD_COLOR)
    small = cv2.resize(voted, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
    for variant in (voted, rotate(voted, 3), 255 - voted, cv2.GaussianBlur(voted, (5, 5), 0), small, jpeg):
        res = ballot.analyze(variant)
        assert res["verdict"] == ballot.VERDICT_VALID, res["boxes"]
        assert res["voted_number"] == 7

    stray = voted.copy()
    cv2.putText(stray, "somchai", (300, img.shape[0] - 40), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.2, PEN, 2)
    res = ballot.analyze(stray)
    assert res["verdict"] == ballot.VERDICT_INVALID and res["strays"]

    double = voted.copy()
    cross(double, boxes[2])
    assert ballot.analyze(double)["verdict"] == ballot.VERDICT_INVALID

    novote = img.copy()
    cross(novote, nv)
    assert ballot.analyze(novote)["verdict"] == ballot.VERDICT_NOVOTE


def test_blank_sheet():
    res = ballot.analyze(np.full((800, 600, 3), 255, np.uint8))
    assert res["verdict"] == ballot.VERDICT_NOBOX and res["counts"]["total"] == 0
