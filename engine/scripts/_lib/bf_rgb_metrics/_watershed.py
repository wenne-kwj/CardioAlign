"""EDM / MaximumFinder / watershed port supplied in rgb_channel_analyzer.py.

The upstream implementation uses SciPy exact EDT and an ImageJ-like watershed;
it has not been verified as bit-identical to ImageJ. Its geometric algorithm is
retained, with a guard for the EDM maximum=1 division-by-zero edge case.
Fluorescence intensities never pass through its internal 8-bit EDM encoding.
"""

UPSTREAM_SHA256 = '6c57e21e627ea9ab541459d3c0fbdfdb83e5a31188494489e80f9b61a054590f'

import numpy as np
from PIL import Image
from scipy import ndimage

# ---------------- ImageJ constants (MaximumFinder.java) ----------------
MAXIMUM, LISTED, PROCESSED = 1, 2, 4
MAX_AREA, EQUAL, MAX_POINT, ELIMINATED = 8, 16, 32, 64
SQRT2 = np.float32(1.4142135624)
TOLERANCE = 0.5                    # MAXFINDER_TOLERANCE in EDM.java
MAX_SORTING_ERROR = float(np.float32(1.1) * SQRT2 / np.float32(2.0))
DIR_X = (0, 1, 1, 1, 0, -1, -1, -1)
DIR_Y = (-1, -1, 0, 1, 1, 1, 0, -1)
DIRECTION_SEQUENCE = (7, 3, 1, 5, 0, 4, 2, 6)

# fate table (makeFateTable), built once
def _make_fate_table():
    table = np.zeros(256, dtype=np.int32)
    for item in range(256):
        isSet = np.array([(item >> i) & 1 for i in range(8)], dtype=bool)
        t = 0
        for i in range(8):
            if isSet[(i + 4) % 8]:
                t |= 1 << i
        for i in range(0, 8, 2):
            if isSet[i]:
                isSet[(i + 1) % 8] = True
                isSet[(i + 7) % 8] = True
        transitions = sum(isSet[i] != isSet[(i + 1) % 8] for i in range(8))
        if transitions >= 4:
            t = 0
        table[item] = t
    return table

FATE_TABLE = _make_fate_table()

U8 = {k: np.uint8(0xFF & ~k) for k in (LISTED, LISTED | EQUAL)}


# ---------------- EDM (exact; ImageJ's approximation differs < 0.09 px) ----------------
def make_edm(mask):
    return ndimage.distance_transform_edt(mask).astype(np.float32)


# ---------------- trueEdmHeight, vectorized ----------------
def true_edm_height(v, w, h):
    """v: float32 EDM (H,W). Returns corrected ridge heights (float32)."""
    trueH = v + np.float32(0.5) * SQRT2
    ridge = np.zeros(v.shape, dtype=bool)
    xmax, ymax = w - 1, h - 1
    y_idx = np.arange(h)[:, None]
    x_idx = np.arange(w)[None, :]
    edge = (x_idx == 0) | (x_idx == xmax) | (y_idx == 0) | (y_idx == ymax)
    for d in range(4):
        d2 = (d + 4) % 8
        dx1, dy1 = DIR_X[d], DIR_Y[d]
        dx2, dy2 = DIR_X[d2], DIR_Y[d2]
        v1 = np.roll(np.roll(v, -dy1, 0), -dx1, 1)   # neighbor at +d
        v2 = np.roll(np.roll(v, -dy2, 0), -dx2, 1)   # neighbor at +d2
        # only valid away from borders; border handled below
        is_ridge = (v >= v1) & (v >= v2)
        hh = np.where(is_ridge, (v1 + v2) / np.float32(2.0), np.minimum(v1, v2))
        hh = hh + (np.float32(1.0) if d % 2 == 0 else SQRT2)
        valid = ~edge
        trueH = np.where(valid & (hh < trueH), hh, trueH)
        ridge |= valid & is_ridge
    out = np.where(edge | (v == 0) | (~ridge), v, trueH)
    return out.astype(np.float32)


# ---------------- getSortedMaxPoints + analyzeAndMarkMaxima ----------------
def _shifted(a, d):
    return np.roll(np.roll(a, -DIR_Y[d], 0), -DIR_X[d], 1)


def find_maxima_types(edm, w, h):
    """Returns (types uint8 array, maxPoints sorted ascending)."""
    v = edm
    globalMin = float(v.min())
    globalMax = float(v.max())
    types = np.zeros(w * h, dtype=np.uint8)
    if globalMax <= globalMin:
        return types, np.empty(0, dtype=np.int64)
    vTrue = true_edm_height(v, w, h)
    yy = np.arange(h)[:, None]
    xx = np.arange(w)[None, :]
    is_max = np.ones((h, w), dtype=bool)
    for d in range(8):
        dx, dy = DIR_X[d], DIR_Y[d]
        vn = np.roll(np.roll(v, -dy, 0), -dx, 1)
        vt = np.roll(np.roll(vTrue, -dy, 0), -dx, 1)
        inb = (xx + dx >= 0) & (xx + dx < w) & (yy + dy >= 0) & (yy + dy < h)
        is_max &= ~(inb & (vn > v) & (vt > vTrue))
    is_max &= (v != globalMin)
    flat = is_max.ravel()
    types[flat] = MAXIMUM
    p = np.nonzero(flat)[0].astype(np.int64)
    fValue = vTrue.ravel()[p]
    vFactor = np.float32(2e9 / (globalMax - globalMin))
    iValue = ((fValue - np.float32(globalMin)) * vFactor).astype(np.int64)
    order = np.lexsort((p, iValue))            # ascending by value, ties by offset
    maxPoints = (iValue[order] << 32) | p[order]
    return types, maxPoints


def analyze_and_mark_maxima(edm, types, maxPoints, w, h):
    """Port of MaximumFinder.analyzeAndMarkMaxima (SEGMENTED, strict=False,
    excludeEdgesNow=False, isEDM=True)."""
    v = edm.ravel()
    vTrue = true_edm_height(edm, w, h).ravel()
    pList = np.empty(w * h, dtype=np.int64)
    tol = np.float32(TOLERANCE)
    mse = np.float32(MAX_SORTING_ERROR)
    for iMax in range(len(maxPoints) - 1, -1, -1):
        offset0 = int(maxPoints[iMax] & 0xFFFFFFFF)
        if types[offset0] & PROCESSED:
            continue
        x0 = offset0 % w
        y0 = offset0 // w
        v0 = vTrue[offset0]
        while True:  # repeat if sortingError
            pList[0] = offset0
            types[offset0] |= (EQUAL | LISTED)
            listLen = 1
            listI = 0
            sortingError = False
            maxPossible = True
            while listI < listLen:
                offset = int(pList[listI])
                x = offset % w
                y = offset // w
                isInner = (y != 0 and y != h - 1) and (x != 0 and x != w - 1)
                for d in range(8):
                    dx, dy = DIR_X[d], DIR_Y[d]
                    if not (isInner or (0 <= x + dx < w and 0 <= y + dy < h)):
                        continue
                    offset2 = offset + (dy * w + dx)
                    if types[offset2] & LISTED:
                        continue
                    if v[offset2] <= 0:
                        continue  # background (non-particles)
                    if types[offset2] & PROCESSED:
                        maxPossible = False
                        break
                    v2 = vTrue[offset2]
                    if v2 > v0 + mse:
                        maxPossible = False
                        break
                    elif v2 >= v0 - tol:
                        if v2 > v0:
                            sortingError = True
                            offset0 = offset2
                            v0 = v2
                            x0 = x + dx
                            y0 = y + dy
                        pList[listLen] = offset2
                        listLen += 1
                        types[offset2] |= LISTED
                        if v2 == v0:
                            types[offset2] |= EQUAL
                if not maxPossible or sortingError:
                    # Java breaks out of the neighbor-direction loop only;
                    # but maxPossible=false also just stops expanding? No:
                    # Java 'break' exits the FOR over directions; the while over
                    # listI continues! Replicate: maxPossible=false -> stop adding,
                    # keep scanning remaining list entries (they only check LISTED
                    # neighbors; PROCESSED neighbor check would re-trigger break).
                    pass
                listI += 1
                if not maxPossible and not sortingError:
                    # after a 'break' due to PROCESSED/higher neighbor, Java still
                    # continues the while(listI<listLen) loop for remaining points.
                    # (no-op: the inner for-loop will break again immediately)
                    pass
            if sortingError:
                for k in range(listLen):
                    types[pList[k]] = 0
            else:
                resetMask = U8[LISTED] if maxPossible else U8[LISTED | EQUAL]
                for k in range(listLen):
                    o = pList[k]
                    types[o] &= resetMask
                    types[o] |= PROCESSED
                    if maxPossible:
                        types[o] |= MAX_AREA
                break  # done with this maximum
    return types


# ---------------- make8bit (EDM branch of MaximumFinder.make8bit) ----------------
def make8bit_edm(edm, types, w, h):
    globalMin = float(edm.min())
    globalMax = float(edm.max())
    minValue = 1.0
    offset = minValue - (globalMax - minValue) * (1.0 / 253 / 2 - 1e-6)
    # A one-pixel-thick foreground has EDM max=1; avoid division by zero.
    factor = min(1.0, 253.0 / (globalMax - minValue)) if globalMax > minValue else 1.0
    raw = edm.ravel()
    out = np.empty(w * h, dtype=np.uint8)
    below = raw < 0.5
    out[below] = 0
    ma = (types & MAX_AREA) != 0
    out[~below & ma] = 255
    rest = ~below & ~ma
    val = 1 + np.floor((raw[rest] - offset) * factor + 0.5)  # Java Math.round
    val = np.clip(val, 1, 254)
    out[rest] = val.astype(np.uint8)
    return out


# ---------------- cleanupMaxima ----------------
def cleanup_maxima(pixels, types, maxPoints, w, h):
    """pixels: uint8 flat (outIp). Port of MaximumFinder.cleanupMaxima."""
    pList = np.empty(w * h, dtype=np.int64)
    for iMax in range(len(maxPoints) - 1, -1, -1):
        offset0 = int(maxPoints[iMax] & 0xFFFFFFFF)
        if types[offset0] & (MAX_AREA | ELIMINATED):
            continue
        level = int(pixels[offset0])
        loLevel = level + 1
        pList[0] = offset0
        types[offset0] |= LISTED
        listLen = 1
        lastLen = 1
        listI = 0
        saddleFound = False
        while not saddleFound and loLevel > 0:
            loLevel -= 1
            lastLen = listLen
            listI = 0
            while listI < listLen and not saddleFound:
                offset = int(pList[listI])
                x = offset % w
                y = offset // w
                isInner = (y != 0 and y != h - 1) and (x != 0 and x != w - 1)
                for d in range(8):
                    dx, dy = DIR_X[d], DIR_Y[d]
                    if not (isInner or (0 <= x + dx < w and 0 <= y + dy < h)):
                        continue
                    offset2 = offset + (dy * w + dx)
                    if types[offset2] & LISTED:
                        continue
                    if (types[offset2] & MAX_AREA) or ((types[offset2] & ELIMINATED) and int(pixels[offset2]) >= loLevel):
                        saddleFound = True
                        break
                    elif int(pixels[offset2]) >= loLevel and not (types[offset2] & ELIMINATED):
                        pList[listLen] = offset2
                        listLen += 1
                        types[offset2] |= LISTED
                listI += 1
        for k in range(listLen):
            types[pList[k]] &= U8[LISTED]
        for k in range(lastLen):
            o = pList[k]
            pixels[o] = loLevel
            types[o] |= ELIMINATED
    return pixels, types


# ---------------- watershedSegment (MaximumFinder.watershedSegment) ----------------
def watershed_segment(pixels, w, h):
    """pixels: uint8 flat. Returns segmented image (255 particles, lines at old values)."""
    hist = np.bincount(pixels, minlength=256)
    arraySize = w * h - hist[0] - hist[255]
    if arraySize <= 0:
        return pixels
    shift = int(np.ceil(np.log2(w)))
    xmask = (1 << shift) - 1
    levelStart = np.zeros(256, dtype=np.int64)
    off = 0
    highestValue = 0
    for vv in range(1, 255):
        levelStart[vv] = off
        off += hist[vv]
        if hist[vv] > 0:
            highestValue = vv
    # fill coordinates sorted by (value, scan order) -- matches Java loop order
    valid = (pixels > 0) & (pixels < 255)
    idx = np.nonzero(valid)[0]
    order = idx[np.argsort(pixels[idx], kind="stable")]
    coordinates = ((order % w) | ((order // w) << shift)).astype(np.int64)
    hist = hist.copy()

    pix = pixels
    for level in range(highestValue, 0, -1):
        remaining = int(hist[level])
        if remaining == 0:
            continue
        idle = 0
        while remaining > 0 and idle < 8:
            sumN = 0
            dIndex = 0
            while True:
                n = _process_level(DIRECTION_SEQUENCE[dIndex % 8], pix, levelStart[level],
                                   remaining, coordinates, w, h, shift, xmask)
                remaining -= n
                sumN += n
                if n > 0:
                    idle = 0
                dIndex += 1
                if not (remaining > 0 and idle < 8):
                    break
                idle += 1
        if remaining > 0 and level > 1:
            nextLevel = level - 1
            while nextLevel > 1 and hist[nextLevel] == 0:
                nextLevel -= 1
            if nextLevel > 0:
                newEnd = int(levelStart[nextLevel] + hist[nextLevel])
                base = levelStart[level]
                for i in range(remaining):
                    xy = int(coordinates[base + i])
                    x = xy & xmask
                    y = (xy & ~xmask) >> shift
                    pOffset = x + y * w
                    add = (x == 0 or y == 0 or x == w - 1 or y == h - 1)
                    if not add:
                        for d in range(8):
                            dx, dy = DIR_X[d], DIR_Y[d]
                            if 0 <= x + dx < w and 0 <= y + dy < h and pix[pOffset + dy * w + dx] == 0:
                                add = True
                                break
                    if add:
                        coordinates[newEnd] = xy
                        newEnd += 1
                hist[nextLevel] = newEnd - int(levelStart[nextLevel])
    return pix


def _process_level(pass_, pix, levelStart, levelNPoints, coordinates, w, h, shift, xmask):
    if levelNPoints == 0:
        return 0
    sl = coordinates[levelStart:levelStart + levelNPoints]
    x = sl & xmask
    y = (sl & ~xmask) >> shift
    o = x + y * w
    idx = np.zeros(levelNPoints, dtype=np.int32)
    m255 = pix == 255
    conds = ((y > 0, o - w, 1),
             ((x < w - 1) & (y > 0), o - w + 1, 2),
             (x < w - 1, o + 1, 4),
             ((x < w - 1) & (y < h - 1), o + w + 1, 8),
             (y < h - 1, o + w, 16),
             ((x > 0) & (y < h - 1), o + w - 1, 32),
             (x > 0, o - 1, 64),
             ((x > 0) & (y > 0), o - w - 1, 128))
    for m, oo, bit in conds:
        idx[m] ^= (bit * m255[oo[m]]).astype(np.int32)
    changed = (FATE_TABLE[idx] & (1 << pass_)) == (1 << pass_)
    if changed.any():
        pix[o[changed]] = 255
    n_changed = int(changed.sum())
    if (~changed).any():
        coordinates[levelStart:levelStart + (levelNPoints - n_changed)] = sl[~changed]
    return n_changed

