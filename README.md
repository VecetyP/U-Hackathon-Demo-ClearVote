# Thai Ballot Verification Engine

Offline OpenCV pipeline that reads a photo or scan of a Thai election ballot (or the EC "บัตรดี / บัตรเสีย" test grid). It finds the voting boxes, classifies each one as `VALID` / `INVALID` / `EMPTY`, flags stray ink outside the boxes, and gives a ruling: **บัตรดี** (with the number voted for), **บัตรเสีย**, or **บัตรไม่ประสงค์ลงคะแนน**.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py      # test bench at http://localhost:8501
pytest -q                 # 32-box benchmark + synthetic ballots
```

## Web app integration (HTTP endpoint)

Use this when the web app isn't Python (e.g. a JavaScript front end or Node back end). It needs no extra dependencies beyond `requirements.txt`.

```bash
git clone https://github.com/VecetyP/U-Hackathon-Demo-ClearVote.git
cd U-Hackathon-Demo-ClearVote
pip install -r requirements.txt
python server.py                    # http://127.0.0.1:8000  (add --host 0.0.0.0 to reach it from other machines)
```

**`POST /evaluate`**: send the ballot image, either as `multipart/form-data` (any file field) or as raw PNG/JPEG bytes. The optional `?first_number=N` is only used when a printed number can't be read. The response is the same object as `ballot.evaluate()`, with the picture base64-encoded:

```json
{
  "output": 25,
  "analysis": {
    "verdict": "valid", "verdict_th": "บัตรดี",
    "reason": "single valid cross in box #25",
    "marks": [{"kind": "candidate_box", "number": 25, "number_source": "printed", "label": "VALID",
               "reason": "single cross inside box", "confidence": 0.95, "bbox": [829, 689, 184, 113]}],
    "boxes_detected": 7, "warnings": [],
    "image": "iVBORw0KGgo..."
  }
}
```

Other responses: `400 {"error": ...}` if the body isn't an image, `413` above 20 MB, and `GET /health` returns `{"ok": true}`. CORS is open, so a browser page on another origin can call it directly.

```js
const form = new FormData();
form.append("image", fileInput.files[0]);                 // or a canvas.toBlob(...) result
const res = await fetch("http://127.0.0.1:8000/evaluate", { method: "POST", body: form });
const { output, analysis } = await res.json();
// output: printed number / 0 (no-vote) / -1 (invalid)
imgEl.src = "data:image/png;base64," + analysis.image;   // analyzed picture
analysis.marks.forEach(m => console.log(m.kind, m.number, m.label, m.reason));
```

```bash
curl -F "image=@ballot.png" http://127.0.0.1:8000/evaluate
```

**Staying up to date:** the API shape above is kept backward compatible, so new fields may be added but existing ones won't be renamed or removed. To pick up pipeline improvements, run `git pull` and restart `python server.py`.

The server has no authentication. It binds to localhost by default; put it behind your app's backend or a reverse proxy before exposing it publicly.

## Integration: one number per ballot (Python)

```python
import cv2, ballot
code = ballot.vote(cv2.imread("ballot.png"))   # or cv2.imdecode(bytes, cv2.IMREAD_COLOR)
```

| Ballot | `vote()` returns |
|---|---|
| Valid, one candidate marked | that candidate's **printed number** (e.g. `25`) |
| Valid, chose not to vote (no-vote box marked, or every box empty) | `0` |
| Invalid (บัตรเสีย) or no ballot found | `-1` |

Need the details too? `ballot.evaluate(img)` returns everything in one object:

```python
obj = ballot.evaluate(img)
obj["output"]                      # main result: number / 0 / -1 (same as vote())
obj["analysis"]["verdict"]         # "valid" | "invalid" | "no_vote" | "no_ballot"  (verdict_th: บัตรดี / บัตรเสีย / ...)
obj["analysis"]["reason"]          # e.g. "2 boxes marked (box #3, box #12); 1 stray mark(s) outside the boxes"
obj["analysis"]["marks"]           # one entry per mark found (empty boxes left out):
#   {"kind": "candidate_box" | "no_vote_box" | "stray", "number": 25, "number_source": "printed" | "position",
#    "label": "VALID" | "INVALID", "reason": "single cross inside box", "confidence": 0.95, "bbox": [x, y, w, h]}
obj["analysis"]["image"]           # analyzed picture (colored boxes + labels) as raw PNG bytes
obj["analysis"]["boxes_detected"], obj["analysis"]["warnings"]
```

`bbox` values are pixel coordinates in the analyzed picture. Images larger than 1600 px on their longest side are scaled down first, so for big photos the coordinates are in the scaled picture, not the original. Everything except `image` is JSON-ready; base64-encode the PNG if you need to send it as JSON.

From the command line: `python ballot.py ballot.png` prints the main code. `ballot.analyze(img)` returns the full result (`output`, `verdict`, per-box labels, strays, overlay image) when you need more than the code.

The candidate number is **read from the print** in the cells left of the X box. A ballot may have up to two cells there, a number cell and a name cell, in either order. They can sit flush against the box or with a gap, and the number can be dark-on-light or light-on-dark. The row is split into printed cells at their border lines. A cell counts as the number cell only if everything printed in it is a confident digit (1–4 digits, no leading zero), so letters in a name that look like digits (B~8, O~0, l~1) are never used. Reading uses an offline digit matcher with templates from common fonts (`digits.npz`, built by `ballot.build_digit_templates()`), so there's no OCR dependency. If no number can be read, it falls back to position (column-major from *First candidate number*), and the app warns you.

## Test bench

The app opens on **Draw on ballot**: the blank EC ballot (`tests/assets/samplevoteFlat.png`) is shown as a canvas. Upload your own (custom) ballot there to draw on it instead. Draw marks with the mouse, press **Analyze** to run them through the pipeline, and use **Undo** / **Clear** to start over. The canvas is a small built-in component in `canvas/index.html`, so there's nothing extra to install. The second tab has file upload, the one-click benchmark and the sample ballots.

## How it works (`ballot.py`)

1. **Normalize**: downscale large photos, auto-invert dark images, and warn on blur.
2. **Box detection**: keep long horizontal and vertical strokes (morphological opening) and turn them into rectangles. A rectangle counts as a box when its interior is mostly blank paper and its size is plausible. Pixel positions are never hard-coded. Three kinds of printed cell are dropped:
   - Filled number cells, because their interior isn't blank.
   - The party-list ballot's logo and party-name cells. They sit flush against the next cell, and the mark box is always the rightmost cell in its row.
   - Name / number label columns on custom ballots: a column whose cells mostly carry black print is labels (a voter marks one box, not most of them).
   - Column headers ("หมายเลขประจำตัวผู้สมัคร", "ช่องทำเครื่องหมาย"). A column holding a single box is a label unless it's the bottom-most box (the no-vote box). The top cells of the columns are compared, and print that repeats across columns is a header. Only top cells are compared, so a voter's two similar marks lower down still void the ballot.

   When a frame mostly contains one inner cell (the no-vote frame: label strip on top, mark area below), the inner mark area is used.
3. **Layout**: the **no-vote box** (ไม่ประสงค์เลือกผู้สมัครผู้ใด) is the box that sits in a column of its own (bottom-right on EC 2569 ballots) or is set apart below the grid. It is classified with the same rules as any other box. The other boxes are numbered column-major. Set *First candidate number* in the app (501 for the EC sample ballot).
4. **Ink per box**: the printed border is erased and strokes that cross it are re-bridged. Ink in the margin counts only if it continues from inside the box. The crop is rescaled to a standard height and at most 1.8:1 width, so an X across a wide box is as steep as one in a candidate box.
5. **Topology classifier**: the mark is skeletonized. For every skeleton point, the classifier counts how many strokes cross a small ring around it.
   - **VALID**: exactly one place where 4 strokes meet (X, +, curved or looped crosses, long tails). That crossing point must be clearly inside the box (at least 10% in from the edges): strokes may run past the border, and that overhang counts as part of the mark rather than a stray. A cross drawn on the border line itself is outside. Sketchy double-stroke crosses get a merged retry.
   - **INVALID**: no crossing (✓ V ^ / ○ □ ♡ ·), a narrow V whose tip fakes a crossing (all its ink lies on one side of that point), overlapping double crosses (6+ arms leave the crossing), more than 4 branches (✱), two or more enclosed regions (☆), several crossings or marks (XX, X X), extra writing next to the cross ((X), X + text), or a mark outside the box.
6. **Stray marks**: ink anywhere on the ballot sheet outside the boxes, found by three detectors:
   - **Vivid pen colour**, for digital pens and bright blue ink.
   - **Faint ballpoint on white paper.** The ink is darker than the paper around it and shifted toward blue in Lab colour space; black print isn't.
   - **Dark freehand marks on coloured panels.** Translucent blue-black ink photographs almost neutral there, so it's told apart from print by shape: pen-width strokes, at least about half a box in size, enclosing paper rather than solid, not part of the box grid, the large dark print or a long printed line.
7. **Verdict**:
   - **บัตรเสีย**: any INVALID box, two or more VALID boxes, or any stray mark.
   - **บัตรดี**: exactly one VALID candidate box.
   - **บัตรไม่ประสงค์ลงคะแนน**: only the no-vote box is marked, or every box is empty.

Every threshold is a named constant at the top of its section in `ballot.py`. Tune them there.

## Known limits

- The digit matcher is tuned for printed sans/serif fonts, Thai UI fonts and OpenCV fonts. On randomized custom ballots (number / name / box in any order, names full of digit-like letters, Thai names) it read about 98% of numbers and never read a wrong one; unread rows fall back to position-based numbering. Decorative or handwritten numbers also fall back.
- A cross needs two strokes that clearly intersect: crossings sharper than about 45°, especially with a thick pen in a small box (phone photos), can be read as invalid. A "V" with a stroke through it at a gentle angle can still pass as a cross. Measured on randomized marks; see the stress numbers in the git history.
- Stray marks drawn over dark print (the title band, the black number columns) are mostly missed, since dark ink on black barely shows. Marks smaller than about half a box on coloured panels, and black-pen marks on white paper, are also missed. A diff against a blank ballot would close these gaps.
- On simulated ballpoint doodles, the detectors catch roughly 95% on white paper and 55–100% on coloured panels, depending on the photo. A photo of a screen is the weakest case. There are zero false strays on all clean test ballots.
- Torn or folded ballots are not detected beyond the "no boxes found" and "blurry" warnings.
