# Thai Ballot Verification Engine

Offline OpenCV pipeline that reads a photo or scan of a Thai election ballot (or the EC "บัตรดี / บัตรเสีย" test grid). It finds the voting boxes, classifies each one as `VALID` / `INVALID` / `EMPTY`, flags stray ink outside the boxes, and gives a ruling: **บัตรดี** (with the number voted for), **บัตรเสีย**, or **บัตรไม่ประสงค์ลงคะแนน**.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py      # test bench at http://localhost:8501
pytest -q                 # 32-box benchmark + synthetic ballots
```

## How it works (`ballot.py`)

1. **Normalize**: downscale large photos, auto-invert dark images, and warn on blur.
2. **Box detection**: keep long horizontal and vertical strokes (morphological opening) and turn them into rectangles. A rectangle counts as a box when its interior is mostly blank paper and its size is plausible. Pixel positions are never hard-coded. Three kinds of printed cell are dropped:
   - Filled number cells, because their interior isn't blank.
   - The party-list ballot's logo and party-name cells. They sit flush against the next cell, and the mark box is always the rightmost cell in its row.
   - Column headers ("หมายเลขประจำตัวผู้สมัคร", "ช่องทำเครื่องหมาย"). Their printed content repeats identically in every column, while hand marks never match pixel-for-pixel.

   When a frame mostly contains one inner cell (the no-vote frame: label strip on top, mark area below), the inner mark area is used.
3. **Layout**: the **no-vote box** (ไม่ประสงค์เลือกผู้สมัครผู้ใด) is the box that sits in a column of its own (bottom-right on EC 2569 ballots) or is set apart below the grid. It is classified with the same rules as any other box. The other boxes are numbered column-major. Set *First candidate number* in the app (501 for the EC sample ballot).
4. **Ink per box**: the printed border is erased and strokes that cross it are re-bridged. Ink in the margin counts only if it continues from inside the box. The crop is rescaled to a standard height and at most 1.8:1 width, so an X across a wide box is as steep as one in a candidate box.
5. **Topology classifier**: the mark is skeletonized. For every skeleton point, the classifier counts how many strokes cross a small ring around it.
   - **VALID**: exactly one place where 4 strokes meet (X, +, curved or looped crosses, long tails). That point must be inside the box, and most of the ink must be inside too. Sketchy double-stroke crosses get a merged retry.
   - **INVALID**: no crossing (✓ V ^ / ○ □ ♡ ·), more than 4 branches (✱), two or more enclosed regions (☆), several crossings or marks (XX, X X), extra writing next to the cross ((X), X + text), or a mark outside the box.
6. **Stray marks**: blue/violet pen ink on the ballot paper, outside every box.
7. **Verdict**:
   - **บัตรเสีย**: any INVALID box, two or more VALID boxes, or any stray mark.
   - **บัตรดี**: exactly one VALID candidate box.
   - **บัตรไม่ประสงค์ลงคะแนน**: only the no-vote box is marked, or every box is empty.

Every threshold is a named constant at the top of its section in `ballot.py`. Tune them there.

## Known limits

- The candidate number comes from the box's position, not from reading the printed digits. The app shows the printed row beside the result so a person can confirm it. Add digit OCR if real ballots skip numbers.
- Stray-mark detection assumes a blue or violet pen. Black-pen strays would need a diff against a blank ballot.
- Torn or folded ballots are not detected beyond the "no boxes found" and "blurry" warnings.
