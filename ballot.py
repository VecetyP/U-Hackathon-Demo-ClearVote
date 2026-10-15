"""Thai ballot verification engine: box detection, mark classification, stray ink, verdict."""
import pathlib
import sys

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
PRINT_MATCH = 0.75      # top cells this similar across columns are printed headers, not marks
LABEL_COLUMN = 0.7       # columns with black print in this share of cells are name/number labels
BLOB_FILL = 0.85        # a white blob filling this much of its bounding box is a borderless box
PAPER_FRAC = 0.55        # interior must be mostly bright paper


def normalize(img):
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    warnings = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    # fallback for photos: a box whose border is lost (e.g. bottom row flush with the end of a dark column)
    # still shows up as a white rectangle; only fill places the line detector left empty
    boxes += [b for b in blob_boxes(paper, H * W)
              if all(iou(b, o) < 0.3 and not contains(o, b, 6) and not contains(b, o, 6) for o in boxes)]
    # drop sub-cells (e.g. quadrants cut by a "+" that spans the box) and small text-like fragments
    # ...but a frame whose inner cell fills most of it is a container (no-vote frame = label strip + mark area):
    # keep the inner mark area instead
    big_inner = lambda o: any(b != o and contains(o, b) and b[2] * b[3] > 0.4 * o[2] * o[3] for b in boxes)
    boxes = [b for b in boxes if not big_inner(b)]
    boxes = [b for b in boxes if not any(o != b and contains(o, b) for o in boxes)]
    if boxes:
        mh = np.median([b[3] for b in boxes])
        boxes = [b for b in boxes if b[3] > 0.6 * mh]
    # party-list ballots: logo / party-name cells are white too, but sit flush against the next cell
    # on the same row; the mark box is always the rightmost cell of its row block
    boxes = [b for b in boxes if not any(
        abs((o[1] + o[3] / 2) - (b[1] + b[3] / 2)) < 0.3 * b[3] and abs(o[0] - (b[0] + b[2])) < 0.1 * b[2]
        for o in boxes if o != b)]
    return drop_label_columns(img, drop_printed(gray, boxes)), raw_lines


def drop_label_columns(img, boxes):
    """Custom ballots may put candidate-number / name cells (white, bordered) left of the X box with a gap.
    A column whose cells mostly carry black print is a label column: a voter marks one box, not most of them."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    printed = lambda b: ((hsv[b[1] + b[3] // 6:b[1] + b[3] * 5 // 6, b[0] + b[2] // 6:b[0] + b[2] * 5 // 6, 2] < 110)
                         & (hsv[b[1] + b[3] // 6:b[1] + b[3] * 5 // 6, b[0] + b[2] // 6:b[0] + b[2] * 5 // 6, 1] < 90)).mean() > 0.01
    drop = set()
    for c in columns(boxes):
        if len(c) >= 3 and sum(map(printed, c)) >= LABEL_COLUMN * len(c):
            drop |= set(c)
    return [b for b in boxes if b not in drop]


def columns(boxes):
    """Group boxes into x-aligned columns, left to right, each sorted top to bottom."""
    mw = np.median([b[2] for b in boxes]) if boxes else 0
    cols = []
    for b in sorted(boxes, key=lambda b: b[0] + b[2] / 2):
        if cols and (b[0] + b[2] / 2) - (cols[-1][-1][0] + cols[-1][-1][2] / 2) < 0.5 * mw:
            cols[-1].append(b)
        else:
            cols.append([b])
    return [sorted(c, key=lambda b: b[1]) for c in cols]


def drop_printed(gray, boxes):
    """Remove column-header cells ("หมายเลขประจำตัวผู้สมัคร", "ช่องทำเครื่องหมาย"):
    - a column holding a single box is a label unless it is the bottom-most box (the no-vote box);
    - the top cells of the columns whose print repeats across columns are headers. Only top cells are compared,
      so a voter's two similar marks lower down still count (and void the ballot)."""
    if len(boxes) < 3:
        return boxes
    lowest = max(boxes, key=lambda b: b[1])
    cols = [c for c in columns(boxes) if len(c) > 1 or c[0] == lowest]
    tops = [c[0] for c in cols]
    sig = []
    for x, y, w, h in tops:
        m = int(0.12 * h)
        c = cv2.resize(gray[y + m:y + h - m, x + m:x + w - m], (64, 40), interpolation=cv2.INTER_AREA).astype(np.float32)
        sig.append(cv2.GaussianBlur(c, (3, 3), 0) if c.std() > 12 else None)
    dup = set()
    for i in range(len(tops)):
        for j in range(i + 1, len(tops)):
            if sig[i] is not None and sig[j] is not None and                     cv2.matchTemplate(sig[i], sig[j][3:-3, 3:-3], cv2.TM_CCOEFF_NORMED).max() > PRINT_MATCH:
                dup |= {tops[i], tops[j]}
    return [b for c in cols for b in c if b not in dup]


def contains(o, b, tol=4):
    return (o[0] - tol <= b[0] and o[1] - tol <= b[1]
            and b[0] + b[2] <= o[0] + o[2] + tol and b[1] + b[3] <= o[1] + o[3] + tol)


def blob_boxes(paper, area):
    """White rectangles (blank paper cells) found as blobs, pen strokes inside closed over first."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(paper.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for x, y, w, h, a in st[1:]:
        if (BOX_ASPECT[0] <= w / h <= BOX_ASPECT[1] and BOX_AREA[0] <= w * h / area <= BOX_AREA[1]
                and a >= BLOB_FILL * w * h):
            out.append((int(x), int(y), int(w), int(h)))
    return out


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
        if all(iou(b, k) < 0.7 for k in keep):  # inner+outer contour of one border
            keep.append(b)
    return keep


BAND = 3          # extra px erased around each detected border line
MARGIN = 0.25     # crop margin around a box, fraction of box height (catches marks over the edge)
EMPTY_PX = 0.002   # total ink below this fraction of box area -> EMPTY
SPECK = 0.004      # components below this fraction of box area are noise (EC accepts a stray dot)
EXTRA_RATIO = 0.2  # extra components summing above this fraction of the main mark -> extra writing
STD_ASPECT = 1.8  # wider boxes are squeezed to this w/h before classification
STD_H = 70         # box crops are rescaled to this height before classification
RING_STEPS = (1, 1.3, 1.6)  # ring radius multipliers tried in order
RING = 0.09        # ring radius for crossing detection, fraction of box height
RETRY_RINGS = (0.14, 0.17, 0.2)  # wider rings tried for the sketchy-stroke retry
INSIDE_MIN = 0.75  # at least this share of the mark's ink must be inside the box
BRANCH_SUPPORT = 0.25  # share of a crossing cluster that must agree on the branch count
HOLE_MIN = 10      # enclosed regions smaller than this (px, standard box) are pen-overlap slivers
MERGE_R = 1.6      # crossing clusters closer than this many ring radii are the same crossing
CLUSTER_MIN = 15   # px in a crossing cluster to count as a real crossing


def ink_mask(gray, boxes, lines):
    """Binary ink with the boxes' printed border lines erased and strokes crossing a border re-bridged."""
    ys = np.concatenate([gray[y:y + h, x:x + w].ravel() for x, y, w, h in boxes])
    paper = np.percentile(ys, 90)
    # ink = clearly darker than its *local* background: a closing wider than a pen stroke erases strokes but keeps
    # region edges, so a coloured band behind a box (e.g. a hot-pink no-vote strip) is background, not ink
    k = max(9, int(0.3 * np.median([h for *_, h in boxes])) | 1)
    bg = np.minimum(cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))), paper)
    ink = (gray.astype(np.int16) < bg.astype(np.int16) - 45).astype(np.uint8) * 255
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
    # normalise to a standard box (height and aspect) so thresholds are in fixed pixels and a corner-to-corner
    # X in a wide box (e.g. the no-vote area) is as steep as in a candidate box
    sy = STD_H / h
    sx = sy * min(1, STD_ASPECT * h / w)
    crop = (cv2.resize(crop, None, fx=sx, fy=sy, interpolation=cv2.INTER_LINEAR) > 127).astype(np.uint8) * 255
    return crop, (round((x - x0) * sx), round((y - y0) * sy), round(w * sx), round(h * sy))


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
    cnts, hier = cv2.findContours(crop, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = sum(1 for cnt, hh in zip(cnts, hier[0]) if hh[3] >= 0 and cv2.contourArea(cnt) >= HOLE_MIN)
    if holes >= 2:  # a cross encloses nothing (a loop-cross encloses one); stars / scribbles enclose several
        return "INVALID", 0.85, "closed shapes (star / scribble)"
    sw = (crop > 0).sum() / max(1, (cv2.ximgproc.thinning(crop) > 0).sum())  # stroke width
    # thin thick strokes first: two fat strokes crossing at a shallow angle overlap over a long stretch,
    # which the skeleton turns into "><" (two forks + a bar) instead of one crossing
    if (it := round((sw - 3) / 2)) > 0:
        thin = cv2.erode(crop, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=it)
        if thin.any():
            crop, sw = thin, sw - 2 * it
    # smallest ring that sees a crossing: thick or shallow crosses need a slightly wider one
    for f in RING_STEPS:
        if clusters := crossing_clusters(crop, f * max(RING * h, 1.2 * sw)):
            break
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
    k = 3
    pts = cv2.dilate(pts, np.ones((k, k), np.uint8))
    n, cl, st, cen = cv2.connectedComponentsWithStats(pts, 8)
    lab = cl[yy[sel], xx[sel]]
    # highest branch count held by a solid share of the cluster: ignores skeleton spurs on thick / blurry strokes
    support = lambda v: int(v.max() if strict else max(u for u in np.unique(v) if (v >= u).mean() >= BRANCH_SUPPORT))
    found = sorted(((st[j, 4], support(c[sel][lab == j]), cen[j][0], cen[j][1])
                    for j in range(1, n) if st[j, 4] >= CLUSTER_MIN), reverse=True)
    # a shallow X lights up a little way along its own arms; fold those into the strongest nearby crossing
    out = []
    for _, br, cx, cy in found:
        near = [o for o in out if np.hypot(o[1] - cx, o[2] - cy) < MERGE_R * R]
        if near:
            near[0][0] = max(near[0][0], br)
        else:
            out.append([br, cx, cy])
    return [tuple(o) for o in out]


def layout(boxes):
    """Split off the no-vote box (isolated below the grid) and order the rest column-major.
    Returns (ordered_candidate_boxes, no_vote_box_or_None)."""
    no_vote = None
    if len(boxes) > 2:
        low = max(boxes, key=lambda b: b[1])
        rest = [b for b in boxes if b != low]
        mh = np.median([b[3] for b in rest])
        above = [b for b in rest if abs((b[0] + b[2] / 2) - (low[0] + low[2] / 2)) < 0.5 * low[2]]
        gaps = np.diff(sorted(b[1] for b in above)) - mh if len(above) > 1 else []
        gap = low[1] - max(b[1] + b[3] for b in rest)
        # set apart below the grid: clearly more space above it than between candidate rows
        isolated_below = gap > 0.8 * mh or (len(gaps) and gap > 1.6 * max(np.median(gaps), 2) and gap > 0.25 * mh)
        # EC 2569 ballots put it bottom-right, beside the grid: a column of its own
        own_column = all(abs((b[0] + b[2] / 2) - (low[0] + low[2] / 2)) > 0.5 * low[2] for b in rest)
        if isolated_below or own_column:
            no_vote, boxes = low, rest
    # ponytail: positional numbering (EC prints every slot sequentially); add digit OCR if ballots skip numbers
    return [b for c in columns(boxes) for b in c], no_vote


def number_strip(img, boxes, b):
    """Row strip left of box b (number / logo / party name) for a human to confirm the printed number."""
    x0, x1 = row_strip(boxes, b, img.shape[1])
    return img[b[1]:b[1] + b[3], x0:x1].copy()


SHEET_PART = 0.05  # light regions this big (fraction of image) are part of the ballot sheet


PEN_SAT = 120  # pen ink saturation that counts anywhere; camera fringes on dark print stay below it


def stray_marks(img, boxes):
    """Pen-colored ink outside every (expanded) voting box -> list of bboxes."""
    # ponytail: assumes a blue/violet pen on a black/green/pink print; black-pen strays need a blank-ballot diff
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # a phone camera adds a desaturated navy fringe around white-on-black print; real pen ink is strongly
    # saturated anywhere, and faint (less saturated) strokes only count on light paper
    light = cv2.medianBlur(val, 31) > 120
    pen = ((hue >= 95) & (hue <= 140) & (val > 60) & (val < 235)  # near-black pixels have noise hue
           & ((sat > PEN_SAT) | ((sat > 70) & light))).astype(np.uint8)
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
    """The ballot sheet: convex hull of the light regions holding boxes plus any large light region
    (e.g. the white header above the black title banner), so strays anywhere on the paper count.
    Small light islands outside the hull (logos / captions around a test infographic) stay out."""
    light = ((hsv[..., 2] > 150) & (hsv[..., 1] < 130)).astype(np.uint8)
    light = cv2.morphologyEx(light, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = SHEET_PART * light.size
    keep = [c for c in cnts if cv2.contourArea(c) >= big or any(
        cv2.pointPolygonTest(c, (float(x + w / 2), float(y + h / 2)), False) >= 0 for x, y, w, h in boxes)]
    mask = np.zeros(light.shape, np.uint8)
    if keep:
        cv2.fillConvexPoly(mask, cv2.convexHull(np.vstack(keep)), 1)
    return mask


def analyze(img, first_number=1):
    img, gray, warnings = normalize(img)
    found, lines = detect_boxes(img, gray)
    if len(found) < 2:
        # maybe a negative (white print on black): use it only if the ballot shows up that way. Brightness
        # alone can't tell - a phone photo with a dark surround is as dark on average as a true negative
        inv, inv_gray = 255 - img, 255 - gray
        inv_found, inv_lines = detect_boxes(inv, inv_gray)
        if len(inv_found) > len(found):
            img, gray, found, lines = inv, inv_gray, inv_found, inv_lines
            warnings.append("inverted colors detected - auto-corrected")
    cands, no_vote = layout(found)
    boxes = cands + ([no_vote] if no_vote else [])
    res = {"boxes": [], "strays": [], "warnings": warnings, "voted_number": None, "number_crop": None}
    if not boxes:
        res.update(verdict=VERDICT_NOBOX, counts=dict(total=0, VALID=0, INVALID=0, EMPTY=0), overlay=img, output=-1)
        res["warnings"].append("no voting boxes found - is this a blank sheet or a ballot photo?")
        return res
    ink = ink_mask(gray, boxes, lines)
    for i, b in enumerate(boxes):
        label, conf, reason = classify(*box_crop(ink, boxes, b))
        number, source = None, None
        if b != no_vote:
            # printed number in the cells left of the box; position-based numbering only as a fallback
            number, _ = read_number(gray, boxes, b)
            number, source = (number, "printed") if number is not None else (first_number + i, "position")
        res["boxes"].append(dict(bbox=b, number=number, number_source=source, is_no_vote=b == no_vote,
                                 label=label, confidence=conf, reason=reason))
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
    res["output"] = {VERDICT_VALID: res["voted_number"], VERDICT_NOVOTE: 0}.get(res["verdict"], -1)
    res["overlay"] = draw(img, res)
    return res


def vote(img, first_number=1):
    """Integration entry point (BGR image, e.g. cv2.imread / cv2.imdecode) -> int:
    the chosen candidate's printed number for a valid ballot, 0 for a valid no-vote ballot
    (no-vote box marked, or every box left empty), -1 for an invalid ballot (บัตรเสีย) or no ballot found."""
    return analyze(img, first_number)["output"]


VERDICT_CODE = {VERDICT_VALID: "valid", VERDICT_INVALID: "invalid", VERDICT_NOVOTE: "no_vote", VERDICT_NOBOX: "no_ballot"}


def evaluate(img, first_number=1):
    """Integration entry point returning everything in one object:
    {"output": int,                      # main result: printed number / 0 (no-vote) / -1 (invalid)
     "analysis": {                       # secondary result
        "verdict": "valid" | "invalid" | "no_vote" | "no_ballot", "verdict_th": "บัตรดี" ...,
        "reason": str,                   # one line: why this verdict
        "marks": [ {...}, ... ],         # one entry per mark found: in-box marks and stray marks
        "boxes_detected": int, "warnings": [str],
        "image": bytes,                  # analyzed picture (overlay), PNG; bboxes are in its pixel coords
     }}"""
    return to_object(analyze(img, first_number))


def to_object(res):
    """analyze() result -> the evaluate() object."""
    marks = [{"kind": "no_vote_box" if b["is_no_vote"] else "candidate_box", "number": b["number"],
              "number_source": b["number_source"], "label": b["label"], "reason": b["reason"],
              "confidence": b["confidence"], "bbox": [int(v) for v in b["bbox"]]}
             for b in res["boxes"] if b["label"] != "EMPTY"]
    marks += [{"kind": "stray", "number": None, "number_source": None, "label": "INVALID",
               "reason": "mark outside the voting boxes (voids the ballot)", "confidence": 0.9,
               "bbox": [int(v) for v in s]} for s in res["strays"]]
    return {"output": res["output"],
            "analysis": {"verdict": VERDICT_CODE[res["verdict"]], "verdict_th": res["verdict"],
                         "reason": verdict_reason(res, marks), "marks": marks,
                         "boxes_detected": res["counts"]["total"], "warnings": res["warnings"],
                         "image": cv2.imencode(".png", res["overlay"])[1].tobytes()}}


def verdict_reason(res, marks):
    tag = lambda m: "no-vote box" if m["kind"] == "no_vote_box" else f"box #{m['number']}"
    if res["verdict"] == VERDICT_NOBOX:
        return "no voting boxes found in the image"
    if res["verdict"] == VERDICT_VALID:
        return f"single valid cross in box #{res['voted_number']}"
    if res["verdict"] == VERDICT_NOVOTE:
        return "no-vote box marked" if marks else "no marks - all boxes empty"
    why = [f"{tag(m)}: {m['reason']}" for m in marks if m["kind"] != "stray" and m["label"] == "INVALID"]
    valid = [m for m in marks if m["label"] == "VALID"]
    if len(valid) > 1:
        why.append(f"{len(valid)} boxes marked ({', '.join(tag(m) for m in valid)})")
    if res["strays"]:
        why.append(f"{len(res['strays'])} stray mark(s) outside the boxes")
    return "; ".join(why)


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


# ---- printed candidate-number reader (offline digit templates, no OCR dependency) ----

DIGITS_FILE = pathlib.Path(__file__).with_name("digits.npz")
DIGIT_SIM = 0.80     # min cosine similarity for a glyph to count as a digit
DIGIT_FONTS = ["arial.ttf", "arialbd.ttf", "ARIALN.TTF", "tahoma.ttf", "tahomabd.ttf", "calibri.ttf", "calibrib.ttf",
               "segoeui.ttf", "segoeuib.ttf", "times.ttf", "timesbd.ttf", "verdana.ttf", "verdanab.ttf",
               "cour.ttf", "courbd.ttf", "consola.ttf", "georgia.ttf", "trebuc.ttf", "LeelawUI.ttf", "LeelaUIb.ttf",
               "leelawad.ttf", "leelawdb.ttf", "angsana.ttc", "cordia.ttc", "browalia.ttc", "upcdl.ttf"]


def digit_vec(mask):
    """Binary glyph -> L2-normalised 24x24 vector (glyph centred in a square, aspect kept)."""
    ys, xs = np.nonzero(mask)
    g = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    s = max(g.shape)
    sq = np.zeros((s, s), np.uint8)
    oy, ox = (s - g.shape[0]) // 2, (s - g.shape[1]) // 2
    sq[oy:oy + g.shape[0], ox:ox + g.shape[1]] = g
    v = cv2.GaussianBlur(cv2.resize(sq, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32), (3, 3), 0).ravel()
    return v / (np.linalg.norm(v) + 1e-6)


def build_digit_templates(font_dir="C:/Windows/Fonts"):
    """One-off: render 0-9 in common ballot fonts + OpenCV Hershey fonts -> digits.npz (committed)."""
    from PIL import Image, ImageDraw, ImageFont
    vecs, labels = [], []

    def add(mask, d):
        for m in (mask, cv2.dilate(mask, np.ones((3, 3), np.uint8)), cv2.erode(mask, np.ones((2, 2), np.uint8))):
            if m.any():
                vecs.append(digit_vec(m))
                labels.append(d)
    for f in DIGIT_FONTS:
        p = pathlib.Path(font_dir) / f
        if not p.exists():
            continue
        font = ImageFont.truetype(str(p), 64)
        for d in range(10):
            im = Image.new("L", (80, 100), 0)
            ImageDraw.Draw(im).text((10, 5), str(d), fill=255, font=font)
            add((np.array(im) > 127).astype(np.uint8) * 255, d)
    for face in (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX):
        for th in (2, 4, 6):
            for d in range(10):
                im = np.zeros((100, 80), np.uint8)
                cv2.putText(im, str(d), (10, 80), face, 2.2, 255, th)
                add(im, d)
    np.savez_compressed(DIGITS_FILE, vecs=np.array(vecs, np.float16), labels=np.array(labels, np.int8))
    return len(vecs)


_TEMPLATES = None


def read_digit(mask):
    """-> (digit, similarity) for one binary glyph."""
    global _TEMPLATES
    if _TEMPLATES is None:
        z = np.load(DIGITS_FILE)
        _TEMPLATES = z["vecs"].astype(np.float32), z["labels"]
    vecs, labels = _TEMPLATES
    sims = vecs @ digit_vec(mask)
    i = int(np.argmax(sims))
    return int(labels[i]), float(sims[i])


def row_strip(boxes, b, W):
    """x-range left of box b up to the previous mark box on the same row (or 6 box widths / page edge)."""
    x, y, w, h = b
    left = [o[0] + o[2] for o in boxes if o[0] + o[2] <= x and abs((o[1] + o[3] / 2) - (y + h / 2)) < 0.5 * h]
    return max(left + [0, int(x - 6 * w)]), x


def read_number(gray, boxes, b):
    """Read the printed candidate number in the cells left of mark box b (number cell and/or name cell,
    any order, dark-on-light or light-on-dark). -> (number, confidence) or (None, 0)."""
    x, y, w, h = b
    x0, x1 = row_strip(boxes, b, gray.shape[1])
    pad = int(0.1 * h)
    strip = gray[max(0, y - pad):y + h + pad, x0:x1]
    if strip.size == 0 or strip.shape[1] < 5:
        return None, 0.0
    glyphs = []
    for fg in (strip < cv2.threshold(strip, 0, 255, cv2.THRESH_OTSU)[0],      # dark print on light cells
               strip > cv2.threshold(strip, 0, 255, cv2.THRESH_OTSU)[0]):     # light print in filled cells
        n, _, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8) * 255, 8)
        lab = cv2.connectedComponents(fg.astype(np.uint8), connectivity=8)[1]
        for i in range(1, n):
            gx, gy, gw, gh, a = st[i]
            touches = gx == 0 or gy == 0 or gx + gw >= strip.shape[1] or gy + gh >= strip.shape[0]
            if touches or not (max(8, 0.18 * h) <= gh <= 0.95 * h) or gw > 1.1 * gh or not 0.15 * gw * gh < a < 0.74 * gw * gh:
                continue  # cell borders, paper, solid blobs (logo discs), specks
            m = (lab[gy:gy + gh, gx:gx + gw] == i).astype(np.uint8) * 255
            hull = cv2.contourArea(cv2.convexHull(cv2.findNonZero(m)))
            if a > 0.9 * hull and gw > 0.5 * gh:
                continue  # solid round blob (logo disc), not a digit; a plain '1' bar is narrow
            glyphs.append((gx, gy, gw, gh, m))
    # the hole of a light '0' shows up as a dark glyph inside it: keep only outermost glyphs
    inside = lambda a, b: a is not b and b[0] <= a[0] and b[1] <= a[1] and a[0] + a[2] <= b[0] + b[2] and a[1] + a[3] <= b[1] + b[3]
    glyphs = sorted((g for g in glyphs if not any(inside(g, o) for o in glyphs)), key=lambda g: g[0])
    # group glyphs into words: a gap wider than ~0.6 glyph height starts a new word (number vs name cell)
    words = []
    for g in glyphs:
        if words and g[0] - (words[-1][-1][0] + words[-1][-1][2]) < 0.6 * g[3] \
                and abs(g[3] - words[-1][-1][3]) < 0.35 * g[3]:
            words[-1].append(g)
        else:
            words.append([g])
    best = (None, 0.0)
    for word in words:
        if len(word) > 4:
            continue
        reads = [read_digit(g[4]) for g in word]
        conf = min(s for _, s in reads)
        if conf >= DIGIT_SIM and (conf > best[1] or best[0] is None):
            best = (int("".join(str(d) for d, _ in reads)), conf)
    return best


if __name__ == "__main__":
    # python ballot.py ballot.png [first_number]  -> prints the vote code (number / 0 / -1)
    if len(sys.argv) < 2:
        sys.exit("usage: python ballot.py <image> [first_number]")
    image = cv2.imread(sys.argv[1])
    if image is None:
        sys.exit(f"cannot read image: {sys.argv[1]}")
    print(vote(image, int(sys.argv[2]) if len(sys.argv) > 2 else 1))
