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
PAPER_V = 130           # min brightness of blank box paper (photos are often shaded toward one edge)
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
    paper = (hsv[..., 2] > PAPER_V) & (hsv[..., 1] < 60)  # unsaturated and fairly bright: shaded photos too
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
CROSS_INSET = 0.1  # the crossing must be this far (fraction of the box) inside its edges; on-the-line = outside
BRANCH_SUPPORT = 0.25  # share of a crossing cluster that must agree on the branch count
HOLE_MIN = 10      # enclosed regions smaller than this (px, standard box) are pen-overlap slivers
ARM_RING = 1.5    # arms are read this many crossing-ring radii out from the crossing
PAIR_TOL = 25     # max deg opposite arms may deviate from a straight line (valid crosses measure <= 18)
ONE_SIDED_GAP = 225  # deg: all far ink inside a <135 deg wedge = strokes meet only at a tip (V); crosses <= ~186
READ_RADII = (0.75, 1, 4 / 3, 1.75)  # radii (x arm ring) for the two-lines / too-many-arms checks
ARM_RADII = (1, 4 / 3)  # arm readings are taken at these multiples of the arm ring; a verdict needs both
TURN_TOL = 60     # max deg an arm may turn between leaving the crossing and its tip (valid <= 31)
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
        # band around the printed border, scaled to the box: a few degrees of tilt (~5% of the width) plus the
        # border itself. A fixed width swallowed half of a small box (and the pen strokes in it) in photos
        cv2.rectangle(band, (x, y), (x + w - 1, y + h - 1), 255, int(0.1 * w) + 8)
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
    if branches > 4 and not two_lines(crop, cx, cy, ARM_RING * max(RING * h, 1.2 * sw)):
        return "INVALID", 0.75, f"{branches} branches (star / asterisk)"
    # where the two lines cross decides it (EC rule): strokes may run past the border, but the crossing must be
    # clearly inside the box - a cross drawn on the border line itself is outside
    mx, my = CROSS_INSET * w, CROSS_INSET * h
    if not (x + mx < cx < x + w - mx and y + my < cy < y + h - my):
        return "INVALID", 0.85, "crossing point outside the box"
    if not sketchy:
        # "two lines clearly intersect": opposite arms must continue each other (rules out K / arrow shapes),
        # and each arm must keep heading away from the crossing (rules out a V corner hanging off it)
        r = ARM_RING * max(RING * h, 1.2 * sw)
        # misalignment must show at two radii: a real K reads consistently, skeleton spurs near the centre of a
        # thick, acute X give one noisy reading
        # read the arms leaving the crossing at several radii and require the evidence to agree:
        # - somewhere, two straight lines must cross (4 of the arms form two opposite pairs; extra arms such as a
        #   tail are fine). A narrow V fakes a crossing near its tip but never shows two lines (2 arms leave it)
        # - an overlapping double cross merges into one "crossing" with 6-8 arms; a single cross never shows 6+
        reads = [arm_angles(crop, cx, cy, k * r) for k in READ_RADII]
        # a V ends at its tip: seen from there, all its far ink lies inside one wedge (one-sided). Any cross - even
        # a sharp-angled one or one with a faint arm - still has ink on opposite sides of the crossing
        far = reads[-2:]
        if (not any(has_two_lines(a) for a in reads)
                and all(a and largest_gap(a) >= ONE_SIDED_GAP for a in far)):
            return "INVALID", 0.8, "not a cross - the strokes only meet at a tip (V / corner shape)"
        if sum(len(a) >= 6 for a in reads) >= 2:
            return "INVALID", 0.8, "more than two lines (double cross / scribble)"
        devs = [opposite_deviation(arm_angles(crop, cx, cy, k * r)) for k in ARM_RADII]
        if all(d is not None and d > PAIR_TOL for d in devs):
            return "INVALID", 0.8, "not two lines crossing (arms don't line up)"
        if any(t > TURN_TOL for t in arm_turns(crop, cx, cy, r)):
            return "INVALID", 0.8, "extra strokes - an arm bends back (not a clean cross)"
    conf = 0.95 - 0.1 * (len(comps) > 1) - 0.1 * sketchy
    return "VALID", round(conf, 2), "single cross inside box"


def two_lines(mask, cx, cy, r):
    """Read further out than the crossing ring, do exactly 4 arms form two straight lines (at both radii)?
    Overrides a noisy branch count near the centre of a thick, shallow X; an asterisk shows 6 arms out here."""
    devs = [opposite_deviation(arm_angles(mask, cx, cy, k * r)) for k in ARM_RADII]
    return all(d is not None and d <= PAIR_TOL for d in devs)


def arm_angles(mask, cx, cy, r):
    """Directions (deg) in which strokes leave (cx, cy), read on a ring of radius r."""
    sk = cv2.dilate(cv2.ximgproc.thinning(mask), np.ones((2, 2), np.uint8)) > 0
    th = np.linspace(0, 2 * np.pi, max(32, int(2 * np.pi * r)), endpoint=False)
    on = sk[np.clip(np.round(cy + r * np.sin(th)).astype(int), 0, mask.shape[0] - 1),
            np.clip(np.round(cx + r * np.cos(th)).astype(int), 0, mask.shape[1] - 1)]
    if on.all() or not on.any():
        return []
    k = int(np.argmin(on))  # start the walk in a gap so no run wraps around
    runs, cur = [], []
    for o, a in zip(np.roll(on, -k), np.roll(th, -k)):
        if o:
            cur.append(a)
        elif cur:
            runs.append(cur)
            cur = []
    runs += [cur] if cur else []
    return [np.degrees(np.angle(np.mean(np.exp(1j * np.array(rn))))) % 360 for rn in runs]


def largest_gap(angles):
    """Largest empty angle (deg) between consecutive arm directions; 360 for a single arm."""
    a = sorted(angles)
    return max([(a[(i + 1) % len(a)] - a[i]) % 360 or 360 for i in range(len(a))])


def has_two_lines(angles):
    """Do some 4 of these arm directions form two straight lines (two opposite pairs within PAIR_TOL)?"""
    from itertools import combinations
    return len(angles) >= 4 and any(opposite_deviation(list(c)) <= PAIR_TOL for c in combinations(angles, 4))


def opposite_deviation(angles):
    """For 4 arms: how far (deg) the worst opposite pair is from a straight line. None if not 4 arms."""
    if len(angles) != 4:
        return None
    a = sorted(angles)
    off = lambda u, v: abs(((v - u) % 360) - 180)
    return max(off(a[0], a[2]), off(a[1], a[3]))


def arm_turns(mask, cx, cy, r):
    """Per arm leaving the crossing: angle (deg) between its start direction and the direction to its tip.
    An arm that returns into the crossing (the loop of a loop-cross) is skipped."""
    sk = cv2.ximgproc.thinning(mask)
    yy, xx = np.nonzero(sk)
    far = np.hypot(xx - cx, yy - cy) > r
    out = np.zeros_like(sk)
    out[yy[far], xx[far]] = 255
    n, lab = cv2.connectedComponents(out, connectivity=8)
    turns = []
    for j in range(1, n):
        ys, xs = np.nonzero(lab == j)
        d = np.hypot(xs - cx, ys - cy)
        near = d < r + 3
        if near.sum() == 0 or len(xs) < 4:
            continue
        start = np.arctan2(ys[near] - cy, xs[near] - cx)
        if np.ptp(start) > 0.6:  # touches the ring in two places: a loop
            continue
        a0 = np.arctan2(ys[near].mean() - cy, xs[near].mean() - cx)
        k = int(np.argmax(d))
        a1 = np.arctan2(ys[k] - cy, xs[k] - cx)
        turns.append(abs(np.degrees((a1 - a0 + np.pi) % (2 * np.pi) - np.pi)))
    return turns


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


SHEET_V = 130     # brightness that counts as ballot paper (any colour); dark surroundings stay out
SHEET_PART = 0.05  # light regions this big (fraction of image) are part of the ballot sheet


OVERHANG = 1.0     # box-heights around a marked box within which its overhanging strokes still belong to it
BALLPOINT_MIN = 0.6  # a ballpoint doodle is at least this many box-heights in both directions
PEN_SAT = 120  # pen ink saturation that counts anywhere; camera fringes on dark print stay below it


def mark_overhang(gray, ink, marked):
    """Ink that continues out of a marked box (a stroke running past the border) belongs to that box's mark, not
    a stray. Limited to about one box-height around the box, so a far doodle can't be pulled in via printed lines.
    Outside the box the stroke may cross darker paper (a green panel in shadow), so a more sensitive ink measure
    is used there; this only ever removes false strays, it never changes a box's label."""
    out = np.zeros(ink.shape, np.uint8)
    if not marked:
        return out
    k = max(9, int(0.3 * np.median([h for *_, h in marked])) | 1)
    paper = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    ink = ((ink > 0) | (gray.astype(np.int16) < paper.astype(np.int16) - 25)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    sizes = st[:, 4]
    for x, y, w, h in marked:
        p = int(0.2 * min(w, h))
        ids = np.unique(lab[y + p:y + h - p, x + p:x + w - p])
        ids = ids[ids > 0]
        if not len(ids):
            continue
        r = int(OVERHANG * h)
        y0, y1, x0, x1 = max(0, y - r), y + h + r, max(0, x - r), x + w + r
        mark = np.isin(lab[y0:y1, x0:x1], ids).astype(np.uint8)
        # a stroke leaving the box often crosses a dark printed border / column edge, where ink is invisible:
        # bridge that small gap, then take the stroke-sized ink it reaches (never big print like a number column)
        g = max(3, int(0.3 * h)) | 1
        reach = cv2.dilate(mark, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g, g)))
        near = [i for i in np.unique(lab[y0:y1, x0:x1][reach > 0]) if i > 0 and sizes[i] <= 2 * w * h]
        out[y0:y1, x0:x1] |= mark | np.isin(lab[y0:y1, x0:x1], near).astype(np.uint8)
    return cv2.dilate(out, np.ones((7, 7), np.uint8))


def stray_marks(img, boxes, lines=None, attached=None):
    """Ink outside every (expanded) voting box -> list of bboxes. Three independent detectors:
    1. vivid pen colour (digital pens, bright blue ink) anywhere on the sheet;
    2. faint ballpoint on white / neutral paper: darker than the paper AND shifted toward blue (black print isn't);
    3. dark freehand marks on coloured paper (green / pink panels), where translucent blue-black ink photographs
       almost neutral: told apart from print by shape (pen-width strokes, at least ~half a box in size, not a solid
       block, not glued to the big dark print, not a printed line) plus a mild blue shift over the whole mark."""
    # ponytail: black-pen strays on *white* paper are still invisible (no colour cue); a blank-ballot diff would fix it
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sheet = paper_mask(hsv, boxes)
    bh = float(np.median([h for *_, h in boxes]))
    grid = np.zeros(img.shape[:2], np.uint8)
    for x, y, w, h in boxes:
        m = int(MARGIN * h)
        grid[max(0, y - m):y + h + m, max(0, x - m):x + w + m] = 1
    if attached is not None:
        grid |= attached  # strokes overhanging from a marked box are part of that mark

    # 1. vivid pen colour. A phone camera adds a desaturated navy fringe around white-on-black print; real pen ink
    #    is strongly saturated anywhere, and fainter strokes only count on light paper
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    light = cv2.medianBlur(val, 31) > 120
    pen = ((hue >= 95) & (hue <= 140) & (val > 60) & (val < 235)  # near-black pixels have noise hue
           & ((sat > PEN_SAT) | ((sat > 70) & light))).astype(np.uint8) & sheet
    pen[grid > 0] = 0
    pen = cv2.morphologyEx(pen, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(pen, 8)
    # real doodles have 2-D extent; JPEG chroma halos along printed edges are 1-4 px thin
    found = [tuple(int(v) for v in st[i, :4]) for i in range(1, n)
             if st[i, 4] >= 0.017 * bh * bh and min(st[i, 2], st[i, 3]) >= 0.15 * bh]

    # 2 + 3. ballpoint ink judged against its local paper colour (Lab; a closing wider than a stroke = paper)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    paper = lambda ch: cv2.medianBlur(cv2.morphologyEx(ch, cv2.MORPH_CLOSE, k), 21).astype(np.int16)
    L, b = lab[..., 0].astype(np.int16), lab[..., 2].astype(np.int16)
    pL, pa, pb = paper(lab[..., 0]), paper(lab[..., 1]) - 128, paper(lab[..., 2])
    darker, bluer = pL - L, pb - b
    neutral = (np.abs(pa) < 12) & (np.abs(pb - 128) < 12) & (pL > 150)
    coloured = (np.hypot(pa, pb - 128) > 18) & (pL > 110)
    # edges of big solid dark print (number columns, title band) are not handwriting
    n, lab_dark, st, _ = cv2.connectedComponentsWithStats((L < 90).astype(np.uint8), 8)
    big = np.isin(lab_dark, [i for i in range(1, n) if st[i, 4] > 2 * bh * bh]).astype(np.uint8)
    keep_out = (cv2.dilate(big, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0) | (grid > 0)
    if lines is not None:
        # printed rules are long; straight stretches of a pen stroke are short and must stay
        n, lab_l, st, _ = cv2.connectedComponentsWithStats(lines, 8)
        long_lines = np.isin(lab_l, [i for i in range(1, n) if max(st[i, 2], st[i, 3]) >= 3 * bh]).astype(np.uint8)
        keep_out |= cv2.dilate(long_lines, np.ones((5, 5), np.uint8)) > 0
    for kind, mask, min_blue in (("white", neutral & (darker > 25) & (bluer > 8), 10),
                                 ("colour", coloured & (darker > 35), 5)):
        mask = (mask & (sheet > 0) & ~keep_out).astype(np.uint8)
        n, lab_i, st, _ = cv2.connectedComponentsWithStats(
            cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))), 8)
        for i in range(1, n):
            x, y, w, h, _ = st[i]
            if not (BALLPOINT_MIN * bh <= min(w, h) and max(w, h) <= 4 * bh):
                continue  # print-sized text, slivers, or sheet-sized shading
            comp = (lab_i[y:y + h, x:x + w] == i) & (mask[y:y + h, x:x + w] > 0)
            px = comp.sum()
            stroke = px / max(1, (cv2.ximgproc.thinning(comp.astype(np.uint8) * 255) > 0).sum())
            hull = cv2.contourArea(cv2.convexHull(cv2.findNonZero(comp.astype(np.uint8)))) + 1
            if (px < 0.017 * bh * bh or stroke < 1.8 or px / (w * h) > 0.45 or px / hull > 0.7
                    or np.median(bluer[y:y + h, x:x + w][comp]) < min_blue):
                continue
            # printed shapes wider than the paper estimate (an arrow, a logo) only show their outline as "darker
            # than paper"; their inside is dark too, while a doodle encloses mostly paper
            hull_mask = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(hull_mask, cv2.convexHull(cv2.findNonZero(comp.astype(np.uint8))), 1)
            ring = np.concatenate([L[y, x:x + w], L[y + h - 1, x:x + w], L[y:y + h, x], L[y:y + h, x + w - 1]])
            if (L[y:y + h, x:x + w][hull_mask > 0] < np.median(ring) - 35).mean() > 0.5:
                continue
            found.append((int(x), int(y), int(w), int(h)))
    return merge_rects(found)


def merge_rects(rects):
    """Union overlapping rectangles (one doodle seen by two detectors is one stray)."""
    rects = list(rects)
    merged = True
    while merged:
        merged = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                a, b = rects[i], rects[j]
                if a[0] <= b[0] + b[2] and b[0] <= a[0] + a[2] and a[1] <= b[1] + b[3] and b[1] <= a[1] + a[3]:
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                    rects[i] = (x0, y0, max(a[0] + a[2], b[0] + b[2]) - x0, max(a[1] + a[3], b[1] + b[3]) - y0)
                    del rects[j]
                    merged = True
                    break
            if merged:
                break
    return rects


def paper_mask(hsv, boxes):
    """The ballot sheet: convex hull of the light regions holding boxes plus any large light region
    (e.g. the white header above the black title banner), so strays anywhere on the paper count.
    Small light islands outside the hull (logos / captions around a test infographic) stay out."""
    # bright of any colour: printed panels (green / pink) are paper too, however saturated the camera makes them
    light = (hsv[..., 2] > SHEET_V).astype(np.uint8)
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
    marked = [r["bbox"] for r in res["boxes"] if r["label"] != "EMPTY"]
    res["strays"] = stray_marks(img, boxes, lines, attached=mark_overhang(gray, ink, marked))
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
GLYPH_MIN_H = 0.12  # printed digits can be small next to big boxes (food ballot: 12 px vs 72 px boxes)
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
    """Read the printed candidate number in the cells left of mark box b. A row may hold just a number cell or a
    number cell and a name cell (either order), flush or with gaps, dark-on-light or light-on-dark print.
    The row strip is split into printed cells at their border lines; a cell is the number cell only if everything
    printed in it (every mark at least 40% as tall as its tallest) is a confident digit - 1 to 4 of them, no leading
    zero. So a dish / party name never contributes, even when its letters look like digits one by one (B~8, O~0, l~1),
    and old-style figures of uneven height (Georgia) are read whole. -> (number, confidence) or (None, 0)."""
    x, y, w, h = b
    x0, x1 = row_strip(boxes, b, gray.shape[1])
    pad = int(0.1 * h)
    strip = gray[max(0, y - pad):y + h + pad, x0:x1]
    if strip.size == 0 or strip.shape[1] < 5:
        return None, 0.0
    thr = cv2.threshold(strip, 0, 255, cv2.THRESH_OTSU)[0]
    marks = []  # (x, y, w, h, mask or None if it can't be a digit)
    for fg in (strip < thr, strip > thr):  # dark print on light cells / light print in filled cells
        n, lab, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
        for i in range(1, n):
            gx, gy, gw, gh, a = st[i]
            touches = gx == 0 or gy == 0 or gx + gw >= strip.shape[1] or gy + gh >= strip.shape[0]
            if touches or not (max(5, 0.05 * h) <= gh <= 0.95 * h):
                continue  # cell borders, specks
            m = (lab[gy:gy + gh, gx:gx + gw] == i).astype(np.uint8) * 255
            hull = cv2.contourArea(cv2.convexHull(cv2.findNonZero(m)))
            disc = a > 0.9 * hull and a > 0.7 * gw * gh and 0.75 < gw / gh < 1.33  # round solid blob (logo)
            if gw > 1.1 * gh or not 0.15 * gw * gh < a < 0.85 * gw * gh or disc:
                if gw <= 1.6 * gh and a < 0.85 * gw * gh:
                    marks.append((gx, gy, gw, gh, None, a / (gw * gh)))  # letter-like, not a digit (wide 'm', logo)
                continue  # blank cell paper (wide and solid) is neither
            marks.append((gx, gy, gw, gh, m, a / (gw * gh)))
    inside = lambda a, b: a is not b and b[0] <= a[0] and b[1] <= a[1] and a[0] + a[2] <= b[0] + b[2] and a[1] + a[3] <= b[1] + b[3]
    # a cell's own border is a hollow frame around its text: never a glyph (and it must not hide the digits in it)
    # (a big '0' also encloses something - its own hole - but it is drawn with solid strokes; a border is thin)
    frame = lambda g: g[3] > 0.5 * h and g[5] < 0.25 and any(inside(o, g) and o[4] is not None for o in marks)
    marks = [g for g in marks if not frame(g)]
    # the hole of a light '0' shows up as a dark mark inside it: keep only outermost marks
    marks = sorted((g for g in marks if not any(inside(g, o) for o in marks)), key=lambda g: g[0])
    best = (None, 0.0)
    for c0, c1 in printed_cells(strip, thr, h):
        cell = [g for g in marks if c0 <= g[0] + g[2] / 2 < c1]
        if not cell:
            continue
        tallest = max(g[3] for g in cell)
        cell = [g for g in cell if g[3] >= 0.4 * tallest]  # drop i-dots / punctuation, keep short old-style digits
        if tallest < max(8, GLYPH_MIN_H * h) or not 1 <= len(cell) <= 4 or any(g[4] is None for g in cell):
            continue
        reads = [read_digit(g[4]) for g in cell]
        conf = min(s for _, s in reads)
        digits = "".join(str(d) for d, _ in reads)
        if conf >= DIGIT_SIM and (len(digits) == 1 or digits[0] != "0") and conf > best[1]:
            best = (int(digits), conf)
    return best


def printed_cells(strip, thr, h):
    """[(x0, x1)] column ranges of the printed cells in a row strip.
    - a thin full-height dark line with light paper on both sides is a cell border (cut there);
    - a wide dark region (a filled number cell with light digits) is a cell of its own. The dark gaps between its
      digits have no paper beside them, so they are not mistaken for borders."""
    col = (strip < thr).mean(0)
    W = len(col)
    k = max(3, int(0.3 * h))
    smooth = np.convolve(col, np.ones(k) / k, mode="same")  # big light digits must not split a filled cell

    def runs(mask):
        out, x = [], 0
        while x < W:
            if mask[x]:
                e = x
                while e < W and mask[e]:
                    e += 1
                out.append((x, e))
                x = e
            else:
                x += 1
        return out
    light = lambda a, b: a < b and col[a:b].mean() < 0.3
    cuts = [(s, e) for s, e in runs(col > 0.6)
            if e - s <= max(3, 0.15 * h) and light(max(0, s - 4), s) and light(e, min(W, e + 4))]
    filled = [(s, e) for s, e in runs(smooth > 0.4) if e - s > 0.5 * h]
    edges = sorted({0, W} | {v for c in cuts for v in c} | {v for f in filled for v in f})
    cut_spans = set(cuts)
    return [(a, b) for a, b in zip(edges, edges[1:]) if b - a > 2 and (a, b) not in cut_spans]


if __name__ == "__main__":
    # python ballot.py ballot.png [first_number]  -> prints the vote code (number / 0 / -1)
    if len(sys.argv) < 2:
        sys.exit("usage: python ballot.py <image> [first_number]")
    image = cv2.imread(sys.argv[1])
    if image is None:
        sys.exit(f"cannot read image: {sys.argv[1]}")
    print(vote(image, int(sys.argv[2]) if len(sys.argv) > 2 else 1))
