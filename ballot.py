"""Thai ballot verification engine: box detection, mark classification, stray ink, verdict."""
import cv2
import numpy as np

GREEN, RED, GRAY = (0, 255, 0), (0, 0, 255), (128, 128, 128)  # BGR for #00FF00 / #FF0000 / #808080
COLORS = {"VALID": GREEN, "INVALID": RED, "EMPTY": GRAY}
VERDICT_VALID, VERDICT_INVALID, VERDICT_NOVOTE = "บัตรดี", "บัตรเสีย", "บัตรไม่ประสงค์ลงคะแนน"
VERDICT_NOBOX = "ไม่พบช่องลงคะแนน"

MAX_SIDE = 1600
LINE_FRAC = 0.035        # min straight-line length for box borders, fraction of image width
BOX_ASPECT = (1.1, 4.0)  # w/h of a voting square
BOX_AREA = (0.0008, 0.05)  # fraction of image area
PAPER_FRAC = 0.55        # interior must be mostly bright paper


def normalize(img):
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    warnings = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 110:
        img, gray = 255 - img, 255 - gray
        warnings.append("inverted colors detected - auto-corrected")
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 60:
        warnings.append("image looks blurry - results may be unreliable")
    return img, gray, warnings


def detect_boxes(img, gray):
    """Rectangles built from long horizontal+vertical strokes with a mostly-blank paper interior."""
    H, W = gray.shape
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    L = max(15, int(W * LINE_FRAC))
    hor = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    ver = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L // 2)))
    raw_lines = hor | ver
    lines = cv2.dilate(raw_lines, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(lines, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    paper = (hsv[..., 2] > 170) & (hsv[..., 1] < 60)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if not (BOX_ASPECT[0] <= w / h <= BOX_ASPECT[1]):
            continue
        if not (BOX_AREA[0] <= w * h / (H * W) <= BOX_AREA[1]):
            continue
        rect = cv2.minAreaRect(c)
        if cv2.contourArea(cv2.convexHull(c)) < 0.85 * rect[1][0] * rect[1][1]:
            continue
        m = max(3, int(min(w, h) * 0.12))
        if paper[y + m:y + h - m, x + m:x + w - m].mean() < PAPER_FRAC:
            continue
        boxes.append((x, y, w, h))
    boxes = dedupe(boxes)
    # drop sub-cells (e.g. quadrants cut by a "+" that spans the box) and small text-like fragments
    boxes = [b for b in boxes if not any(o != b and contains(o, b) for o in boxes)]
    if boxes:
        mh = np.median([b[3] for b in boxes])
        boxes = [b for b in boxes if b[3] > 0.6 * mh]
    # party-list ballots: logo / party-name cells are white too, but sit flush against the next cell
    # on the same row; the mark box is always the rightmost cell of its row block
    boxes = [b for b in boxes if not any(
        abs((o[1] + o[3] / 2) - (b[1] + b[3] / 2)) < 0.3 * b[3] and abs(o[0] - (b[0] + b[2])) < 0.1 * b[2]
        for o in boxes if o != b)]
    return boxes, raw_lines


def contains(o, b, tol=4):
    return (o[0] - tol <= b[0] and o[1] - tol <= b[1]
            and b[0] + b[2] <= o[0] + o[2] + tol and b[1] + b[3] <= o[1] + o[3] + tol)


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter)


def dedupe(boxes):
    boxes = sorted(boxes, key=lambda b: -b[2] * b[3])
    keep = []
    for b in boxes:
        if all(iou(b, k) < 0.5 for k in keep):
            keep.append(b)
    return keep


BAND = 3          # extra px erased around each detected border line
MARGIN = 0.25     # crop margin around a box, fraction of box height (catches marks over the edge)
EMPTY_PX = 0.002   # total ink below this fraction of box area -> EMPTY
SPECK = 0.004      # components below this fraction of box area are noise (EC accepts a stray dot)
EXTRA_RATIO = 0.2  # extra components summing above this fraction of the main mark -> extra writing
STD_H = 70         # box crops are rescaled to this height before classification
RING = 0.09        # ring radius for crossing detection, fraction of box height
RETRY_RINGS = (0.14, 0.17, 0.2)  # wider rings tried for the sketchy-stroke retry
INSIDE_MIN = 0.75  # at least this share of the mark's ink must be inside the box
BRANCH_SUPPORT = 0.25  # share of a crossing cluster that must agree on the branch count
CLUSTER_MIN = 15   # px in a crossing cluster to count as a real crossing


def ink_mask(gray, boxes, lines):
    """Binary ink with the boxes' printed border lines erased and strokes crossing a border re-bridged."""
    ys = np.concatenate([gray[y:y + h, x:x + w].ravel() for x, y, w, h in boxes])
    paper = np.percentile(ys, 90)
    ink = (gray < paper - 45).astype(np.uint8) * 255
    band = np.zeros_like(ink)
    for x, y, w, h in boxes:
        cv2.rectangle(band, (x, y), (x + w - 1, y + h - 1), 255, 8 * BAND)  # wide enough for a few degrees of tilt
    erase = cv2.dilate(lines, np.ones((7, 7), np.uint8)) & band
    ink[erase > 0] = 0
    bridged = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    return np.where(erase > 0, bridged, ink)


def crossings(sk, R):
    """For each skeleton pixel, how many strokes cross a circle of radius R around it."""
    n = max(16, int(2 * np.pi * R))
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ox, oy = np.round(R * np.cos(t)).astype(int), np.round(R * np.sin(t)).astype(int)
    skd = cv2.dilate(sk, np.ones((2, 2), np.uint8)) > 0
    H, W = sk.shape
    yy, xx = np.nonzero(sk)
    s = skd[np.clip(yy[:, None] + oy, 0, H - 1), np.clip(xx[:, None] + ox, 0, W - 1)].astype(np.int8)
    return xx, yy, ((s - np.roll(s, 1, axis=1)) == 1).sum(1)


def box_crop(ink, boxes, b):
    """Ink around box b (with margin), other boxes blanked out. Returns crop and its origin."""
    x, y, w, h = b
    m = int(MARGIN * h)
    x0, y0 = max(0, x - m), max(0, y - m)
    crop = ink[y0:y + h + m, x0:x + w + m].copy()
    for o in boxes:
        if o != b:
            ox, oy, ow, oh = o
            cv2.rectangle(crop, (ox - x0 - 2, oy - y0 - 2), (ox - x0 + ow + 2, oy - y0 + oh + 2), 0, -1)
    # margin ink only counts if it continues from inside the box (neighbouring print stays out)
    n, lab = cv2.connectedComponents(crop, connectivity=8)
    p = 2 * BAND + 2
    inner = lab[y - y0 + p:y - y0 + h - p, x - x0 + p:x - x0 + w - p]
    crop[~np.isin(lab, np.unique(inner[inner > 0]))] = 0
    # normalise scale so every threshold below is in "standard box" pixels
    s = STD_H / h
    crop = (cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR) > 127).astype(np.uint8) * 255
    return crop, (round((x - x0) * s), round((y - y0) * s), round(w * s), round(h * s))


def classify(crop, rect):
    """-> (label, confidence, reason). Topology rules for a Thai EC valid cross; rect = box in crop coords."""
    x, y, w, h = rect
    area = w * h
    if (crop > 0).sum() < EMPTY_PX * area:
        return "EMPTY", 0.99, "no ink"
    n, lab, st, _ = cv2.connectedComponentsWithStats(crop, 8)
    comps = sorted((st[i, 4] for i in range(1, n) if st[i, 4] >= SPECK * area), reverse=True)
    if not comps:
        return "INVALID", 0.7, "dot / tiny mark"
    if len(comps) > 1 and comps[1] > 0.5 * comps[0]:
        return "INVALID", 0.85, "multiple marks in one box"
    if sum(comps[1:]) > EXTRA_RATIO * comps[0]:
        return "INVALID", 0.85, "extra marks or writing"
    sw = (crop > 0).sum() / max(1, (cv2.ximgproc.thinning(crop) > 0).sum())  # stroke width
    clusters = crossing_clusters(crop, max(RING * h, 1.2 * sw))
    if len(clusters) > 1:
        # sketchy pens: double/dashed strokes create near-duplicate crossings -> merge and look wider
        merged = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        retry = [r for f in RETRY_RINGS if len(r := crossing_clusters(merged, f * h, strict=True)) == 1 and r[0][0] == 4]
        if not retry:
            return "INVALID", 0.8, "multiple crossings / marks"
        clusters, sketchy = retry[0], True
    else:
        sketchy = False
    if not clusters:
        return "INVALID", 0.85, "no crossing (not an X / +)"
    branches, cx, cy = clusters[0]
    if branches > 4:
        return "INVALID", 0.75, f"{branches} branches (star / asterisk)"
    inside = crop[y:y + h, x:x + w]
    if not (x < cx < x + w and y < cy < y + h) or (inside > 0).sum() < INSIDE_MIN * (crop > 0).sum():
        return "INVALID", 0.85, "mark outside the box"
    conf = 0.95 - 0.1 * (len(comps) > 1) - 0.1 * sketchy
    return "VALID", round(conf, 2), "single cross inside box"


def crossing_clusters(mask, R, strict=False):
    """[(branches, cx, cy)] for each place where >=4 strokes meet, in mask coords.
    strict: branches = max seen in the cluster (used by the lenient sketchy retry)."""
    sk = cv2.ximgproc.thinning(mask)
    xx, yy, c = crossings(sk, max(3.5, R))
    sel = c >= 4
    pts = np.zeros_like(sk)
    pts[yy[sel], xx[sel]] = 255
    pts = cv2.dilate(pts, np.ones((3, 3), np.uint8))
    n, cl, st, cen = cv2.connectedComponentsWithStats(pts, 8)
    lab = cl[yy[sel], xx[sel]]
    # highest branch count held by a solid share of the cluster: ignores skeleton spurs on thick / blurry strokes
    support = lambda v: int(v.max() if strict else max(u for u in np.unique(v) if (v >= u).mean() >= BRANCH_SUPPORT))
    return [(support(c[sel][lab == j]), cen[j][0], cen[j][1]) for j in range(1, n) if st[j, 4] >= CLUSTER_MIN]


def layout(boxes):
    """Split off the no-vote box (isolated below the grid) and order the rest column-major.
    Returns (ordered_candidate_boxes, no_vote_box_or_None)."""
    no_vote = None
    if len(boxes) > 2:
        low = max(boxes, key=lambda b: b[1])
        rest = [b for b in boxes if b != low]
        mh = np.median([b[3] for b in rest])
        if low[1] - max(b[1] + b[3] for b in rest) > 0.8 * mh:
            no_vote, boxes = low, rest
    mw = np.median([b[2] for b in boxes]) if boxes else 0
    cols = []
    for b in sorted(boxes, key=lambda b: b[0] + b[2] / 2):
        if cols and (b[0] + b[2] / 2) - (cols[-1][-1][0] + cols[-1][-1][2] / 2) < 0.5 * mw:
            cols[-1].append(b)
        else:
            cols.append([b])
    # ponytail: positional numbering (EC prints every slot sequentially); add digit OCR if ballots skip numbers
    return [b for c in cols for b in sorted(c, key=lambda b: b[1])], no_vote


def number_strip(img, boxes, b):
    """Row strip left of box b (number / logo / party name) for a human to confirm the printed number."""
    x, y, w, h = b
    left = [o[0] + o[2] for o in boxes if o[0] + o[2] <= x and abs(o[1] - y) < 0.5 * h]
    x0 = max(left + [0, int(x - 4 * w)])
    return img[y:y + h, x0:x].copy()


def stray_marks(img, boxes):
    """Pen-colored ink outside every (expanded) voting box -> list of bboxes."""
    # ponytail: assumes a blue/violet pen on a black/green/pink print; black-pen strays need a blank-ballot diff
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pen = ((hsv[..., 0] >= 95) & (hsv[..., 0] <= 140) & (hsv[..., 1] > 70) & (hsv[..., 2] < 235)).astype(np.uint8)
    pen &= paper_mask(hsv, boxes)
    for x, y, w, h in boxes:
        m = int(MARGIN * h)
        pen[max(0, y - m):y + h + m, max(0, x - m):x + w + m] = 0
    pen = cv2.morphologyEx(pen, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(pen, 8)
    bh = np.median([h for *_, h in boxes])
    # real doodles have 2-D extent; JPEG chroma halos along printed edges are 1-4 px thin
    return [tuple(int(v) for v in st[i, :4]) for i in range(1, n)
            if st[i, 4] >= 0.01 * bh * bh * 1.7 and min(st[i, 2], st[i, 3]) >= 0.15 * bh]


def paper_mask(hsv, boxes):
    """The light sheet(s) the boxes sit on (white / light-green / pink paper), holes filled."""
    light = ((hsv[..., 2] > 150) & (hsv[..., 1] < 130)).astype(np.uint8)
    light = cv2.morphologyEx(light, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(light.shape, np.uint8)
    for c in cnts:
        if any(cv2.pointPolygonTest(c, (float(x + w / 2), float(y + h / 2)), False) >= 0 for x, y, w, h in boxes):
            cv2.drawContours(mask, [c], -1, 1, -1)
    return mask


def analyze(img, first_number=1):
    img, gray, warnings = normalize(img)
    found, lines = detect_boxes(img, gray)
    cands, no_vote = layout(found)
    boxes = cands + ([no_vote] if no_vote else [])
    res = {"boxes": [], "strays": [], "warnings": warnings, "voted_number": None, "number_crop": None}
    if not boxes:
        res.update(verdict=VERDICT_NOBOX, counts=dict(total=0, VALID=0, INVALID=0, EMPTY=0), overlay=img)
        res["warnings"].append("no voting boxes found - is this a blank sheet or a ballot photo?")
        return res
    ink = ink_mask(gray, boxes, lines)
    for i, b in enumerate(boxes):
        label, conf, reason = classify(*box_crop(ink, boxes, b))
        res["boxes"].append(dict(bbox=b, number=None if b == no_vote else first_number + i,
                                 is_no_vote=b == no_vote, label=label, confidence=conf, reason=reason))
    res["strays"] = stray_marks(img, boxes)
    counts = {k: sum(r["label"] == k for r in res["boxes"]) for k in COLORS}
    res["counts"] = dict(total=len(boxes), **counts)
    valid = [r for r in res["boxes"] if r["label"] == "VALID"]
    if counts["INVALID"] or len(valid) > 1 or res["strays"]:
        res["verdict"] = VERDICT_INVALID
    elif not valid or valid[0]["is_no_vote"]:
        res["verdict"] = VERDICT_NOVOTE
    else:
        res["verdict"] = VERDICT_VALID
        res["voted_number"] = valid[0]["number"]
        res["number_crop"] = number_strip(img, boxes, valid[0]["bbox"])
    res["overlay"] = draw(img, res)
    return res


def draw(img, res):
    out = img.copy()
    for r in res["boxes"]:
        x, y, w, h = r["bbox"]
        col = COLORS[r["label"]]
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 3)
        tag = ("NO-VOTE" if r["is_no_vote"] else f"#{r['number']}") + f" {r['label']} {r['confidence']:.2f}"
        fs = max(0.35, h / 160)
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        cv2.rectangle(out, (x, y - th - 6), (x + tw + 4, y), col, -1)
        cv2.putText(out, tag, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)
    for x, y, w, h in res["strays"]:
        cv2.rectangle(out, (x - 4, y - 4), (x + w + 4, y + h + 4), RED, 2)
        cv2.putText(out, "STRAY", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1, cv2.LINE_AA)
    return out
