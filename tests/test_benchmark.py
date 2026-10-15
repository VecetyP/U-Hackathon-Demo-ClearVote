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


# ---- real EC municipal ballot (green), blank scan ----

REAL = pathlib.Path(__file__).parent / "assets" / "samplevoteFlat.png"


def real_boxes():
    """Box rects in original-image pixels, keyed by printed number ('nv' = no-vote)."""
    img = cv2.imread(str(REAL))
    res = ballot.analyze(img, first_number=501)
    s = max(img.shape[:2]) / ballot.MAX_SIDE
    return img, res, {("nv" if b["is_no_vote"] else b["number"]): tuple(int(v * s) for v in b["bbox"])
                      for b in res["boxes"]}


def big_cross(img, box):
    x, y, w, h = box
    cv2.line(img, (x + w // 5, y + h // 5), (x + 4 * w // 5, y + 4 * h // 5), PEN, 5)
    cv2.line(img, (x + w // 5, y + 4 * h // 5), (x + 4 * w // 5, y + h // 5), PEN, 5)


def test_real_blank_ballot_ignores_printed_headers():
    _, res, boxes = real_boxes()
    assert sorted(k for k in boxes if k != "nv") == list(range(501, 521))
    assert "nv" in boxes, "no-vote mark area (bottom right) must be found"
    assert all(b["label"] == "EMPTY" for b in res["boxes"]) and not res["strays"]
    assert res["verdict"] == ballot.VERDICT_NOVOTE


def test_real_ballot_marks():
    img, _, boxes = real_boxes()
    one = img.copy()
    big_cross(one, boxes[507])
    for v in (one, cv2.resize(one, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)):  # low-res: headers stay out
        res = ballot.analyze(v, first_number=501)
        assert res["verdict"] == ballot.VERDICT_VALID and res["voted_number"] == 507 and res["output"] == 507
        assert res["counts"]["total"] == 21

    nv = img.copy()
    big_cross(nv, boxes["nv"])
    assert ballot.analyze(nv, first_number=501)["verdict"] == ballot.VERDICT_NOVOTE

    both = one.copy()
    big_cross(both, boxes["nv"])
    assert ballot.analyze(both, first_number=501)["verdict"] == ballot.VERDICT_INVALID

    check = img.copy()
    x, y, w, h = boxes[512]
    cv2.polylines(check, [np.array([(x + 20, y + h // 2), (x + w // 3, y + h - 20), (x + w - 15, y + 15)])], False, PEN, 5)
    assert ballot.analyze(check, first_number=501)["verdict"] == ballot.VERDICT_INVALID

    for key in (507, "nv"):  # EC-invalid symbols, in a candidate box and in the wide no-vote area
        x, y, w, h = boxes[key]
        c = np.array([x + w / 2, y + h / 2])
        star = np.int32([c + h / 3 * np.array([np.sin(a), -np.cos(a)]) for a in np.arange(5) * 4 * np.pi / 5])
        shapes = [lambda v: cv2.polylines(v, [star], True, PEN, 4),
                  lambda v: cv2.circle(v, tuple(np.int32(c)), h // 4, PEN, 5),
                  lambda v: cv2.line(v, (x + w // 4, y + 3 * h // 4), (x + 3 * w // 4, y + h // 4), PEN, 5),
                  lambda v: [cv2.line(v, tuple(np.int32(c - h / 3 * d)), tuple(np.int32(c + h / 3 * d)), PEN, 4)
                             for d in (np.array([np.cos(a), np.sin(a)]) for a in np.deg2rad([0, 60, 120]))]]
        for i, draw in enumerate(shapes):
            v = img.copy()
            draw(v)
            assert ballot.analyze(v, first_number=501)["verdict"] == ballot.VERDICT_INVALID, (key, i)

    stray = one.copy()
    cv2.putText(stray, "somchai", (150, 1900), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, PEN, 4)
    res = ballot.analyze(stray, first_number=501)
    assert res["verdict"] == ballot.VERDICT_INVALID and res["strays"]

    header = one.copy()  # doodle over the white header / title banner, above the green panel
    cv2.ellipse(header, (390, 540), (130, 45), 0, 0, 360, PEN, 5)
    res = ballot.analyze(header, first_number=501)
    assert res["verdict"] == ballot.VERDICT_INVALID and res["strays"]


# ---- custom ballots: printed numbers are not sequential, name/number cells left of the X box ----

CUSTOM = {"custom_ballot_gaps.png": [3, 7, 12, 25, 31, 48], "custom_ballot_flush.png": [101, 5, 77, 9, 60, 2]}
NAMES = ["นายสมชาย ใจดี", "นางสาวมาลี รักไทย", "นายประยุทธ์ มั่นคง", "นางวันดี สุขใจ", "นายอนันต์ ทองคำ", "นายกิตติ ศรีสุข"]


def make_custom_ballots(font_dir="C:/Windows/Fonts"):
    """One-off generator for the committed custom ballot assets (needs Windows Thai fonts)."""
    from PIL import Image, ImageDraw, ImageFont
    thai = ImageFont.truetype(f"{font_dir}/LeelawUI.ttf", 34)
    num = ImageFont.truetype(f"{font_dir}/arialbd.ttf", 46)
    for fname, numbers in CUSTOM.items():
        flush = "flush" in fname
        im = Image.new("RGB", (1200, 1500), "white")
        d = ImageDraw.Draw(im)
        d.text((330, 50), "บัตรเลือกตั้ง (ตัวอย่าง)", font=ImageFont.truetype(f"{font_dir}/LeelaUIb.ttf", 48), fill="black")
        d.text((120, 140), "ทำเครื่องหมาย X ในช่องทำเครื่องหมาย ได้เพียงช่องเดียว", font=thai, fill="black")
        for r, (n, name) in enumerate(zip(numbers, NAMES)):
            y = 240 + r * 150
            cells = [("name", 470), ("num", 160)] if flush else [("num", 160), ("name", 470)]
            x = 120
            for kind, cw in cells:
                d.rectangle([x, y, x + cw, y + 110], outline="black", width=4)
                if kind == "num":
                    d.text((x + cw // 2, y + 55), str(n), font=num, fill="black", anchor="mm")
                else:
                    d.text((x + 20, y + 55), name, font=thai, fill="black", anchor="lm")
                x += cw + (0 if flush else 40)
            d.rectangle([x, y, x + 180, y + 110], outline="black", width=4)
        y = 240 + len(numbers) * 150 + 40
        d.text((120, y + 35), "ไม่ประสงค์ลงคะแนน", font=thai, fill="black")
        d.rectangle([840, y, 1020, y + 110], outline="black", width=4)
        im.save(pathlib.Path(__file__).parent / "assets" / fname)



@pytest.mark.parametrize("fname", CUSTOM)
def test_custom_ballot_output_codes(fname):
    numbers = CUSTOM[fname]
    img = cv2.imread(str(pathlib.Path(__file__).parent / "assets" / fname))
    res = ballot.analyze(img)
    assert res["counts"]["total"] == len(numbers) + 1, "name / number cells must not count as mark boxes"
    assert [b["number"] for b in res["boxes"] if not b["is_no_vote"]] == numbers  # read from print, not position
    assert all(b["number_source"] == "printed" for b in res["boxes"] if not b["is_no_vote"])
    assert res["output"] == 0  # blank ballot = no-vote
    boxes = {b["number"] if not b["is_no_vote"] else "nv": b["bbox"] for b in res["boxes"]}

    for n in numbers:
        v = img.copy()
        big_cross(v, boxes[n])
        assert ballot.vote(v) == n
    v = img.copy()
    big_cross(v, boxes["nv"])
    assert ballot.vote(v) == 0
    v = img.copy()
    big_cross(v, boxes[numbers[0]])
    big_cross(v, boxes[numbers[3]])
    assert ballot.vote(v) == -1  # two candidates
    v = img.copy()
    big_cross(v, boxes[numbers[1]])
    big_cross(v, boxes["nv"])
    assert ballot.vote(v) == -1  # candidate + no-vote
    v = img.copy()
    x, y, w, h = boxes[numbers[2]]
    cv2.polylines(v, [np.array([(x + 25, y + h // 2), (x + w // 3, y + h - 20), (x + w - 20, y + 20)])], False, PEN, 5)
    assert ballot.vote(v) == -1  # check mark
    v = img.copy()
    big_cross(v, boxes[numbers[4]])
    cv2.putText(v, "hello", (150, 1400), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.5, PEN, 3)
    assert ballot.vote(v) == -1  # stray writing


if __name__ == "__main__":
    make_custom_ballots()
