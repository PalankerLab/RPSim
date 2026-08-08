"""
One-time offline tool: builds diode_side_PS100-lg.pkl, a per-pixel lookup (same shape and
indexing as pixel_label_PS100-lg.pkl) marking which of the two photodiodes each hexagon's
pixel belongs to (0 = not a diode, 1 or 2 = which half).

Each bipolar PRIMA 100-lg hexagon is split into two photodiodes by a thin divider visible in
Grid_PS100-lg_with_diodes.png. calibrate_split_angles() measures that divider's direction
straight off the photo (registered onto the pixel_label grid) and shows it alternates by row
between two fixed, mirror-image angles -- not a per-hexagon, photo-dependent property. So the
photo is only needed once, to calibrate those two angles; build_diode_side_labels() then does
a purely geometric split using pixel_labels alone.

Run directly to (re)generate the file:
    python -m utilities.generate_diode_side_labels
"""
import pickle
import numpy as np
from PIL import Image
from scipy import ndimage

IMAGE_SEQUENCE_DIR = "user_files/user_input/image_sequence"
PIXEL_LABEL_FILE = f"{IMAGE_SEQUENCE_DIR}/pixel_label_PS100-lg.pkl"
DIODE_PHOTO_FILE = f"{IMAGE_SEQUENCE_DIR}/Grid_PS100-lg_with_diodes.png"
OUTPUT_FILE = f"{IMAGE_SEQUENCE_DIR}/diode_side_PS100-lg.pkl"

# Grid_PS100-lg_with_diodes.png ships rotated 90 degrees and mirrored relative to
# Grid_PS100-lg.png / pixel_label_PS100-lg.pkl. The rotation was confirmed both visually and via
# the lattice's FFT period ratio (1.150 for the label grid vs. 1.147 for the photo once rotated).
# The mirror is needed on top of that: a hexagonal lattice's positions are symmetric under
# reflection, so a registration fit from centroids alone can lock onto a tight-residual match
# with or without the flip -- the residual metric can't tell them apart by itself. Comparing all
# 8 rotation/flip combinations by how many hexagons they match (not just residual) breaks the
# tie: rotate 90 + horizontal flip matches all 378 hexagons at ~0.37px, vs. 365/378 with no flip.
DIODE_PHOTO_ROTATION_DEG = 90
DIODE_PHOTO_FLIP = Image.FLIP_LEFT_RIGHT


def _hexagon_centroids(pixel_labels):
    labels = np.unique(pixel_labels)
    labels = labels[labels != 0]
    centroids = np.array(ndimage.center_of_mass(np.ones_like(pixel_labels), pixel_labels, labels))
    return labels, centroids  # centroids columns are (y, x)


def _register_diode_photo(pixel_labels, photo_path):
    """
    Registers the (rotated) diode photo onto pixel_labels' pixel grid.
    The two renderings differ only by rotation + isotropic scale + translation (no shear/skew
    once rotated -- verified by fitting a full affine and finding negligible off-diagonal terms),
    so a scale+translate fit from matched hexagon/diode centroids is sufficient.
    """
    labels, hex_centroids = _hexagon_centroids(pixel_labels)
    hex_xy = hex_centroids[:, ::-1]

    photo = Image.open(photo_path).convert("RGB").rotate(DIODE_PHOTO_ROTATION_DEG, expand=True)
    if DIODE_PHOTO_FLIP is not None:
        photo = photo.transpose(DIODE_PHOTO_FLIP)
    arr = np.array(photo).astype(int)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    lum = r + g + b
    # The diode's metal contact: a small brownish-grey circle, distinct from the bright green
    # photodiode fill (much higher luminance) and the very dark hexagon border (much lower).
    diode_mask = (lum > 250) & (lum < 420) & (r >= g - 10) & (np.abs(r - b) < 40)
    blob_labels, n_blobs = ndimage.label(diode_mask)
    sizes = ndimage.sum(diode_mask, blob_labels, range(1, n_blobs + 1))
    good_blobs = np.flatnonzero((sizes > 15) & (sizes < 150)) + 1
    diode_centroids = np.array(ndimage.center_of_mass(diode_mask, blob_labels, good_blobs))
    diode_xy = diode_centroids[:, ::-1]

    # Coarse isotropic scale+translate from median nearest-neighbor spacing, to get an initial
    # correspondence between hexagon centroids and diode blobs.
    def median_nn_dist(pts):
        from scipy.spatial import cKDTree
        d, _ = cKDTree(pts).query(pts, k=2)
        return np.median(d[:, 1])

    scale0 = median_nn_dist(hex_xy) / median_nn_dist(diode_xy)
    translate0 = hex_xy.mean(axis=0) - diode_xy.mean(axis=0) * scale0

    from scipy.spatial import cKDTree
    diode_xy_coarse = diode_xy * scale0 + translate0
    dist, nn_idx = cKDTree(diode_xy_coarse).query(hex_xy, k=1)
    matched = dist < median_nn_dist(hex_xy) / 2

    # Refine with a full least-squares affine fit on the matched pairs (this is where we'd
    # absorb any small residual rotation/shear, though in practice it comes out negligible).
    src = diode_xy[nn_idx[matched]]
    dst = hex_xy[matched]
    X = np.hstack([src, np.ones((src.shape[0], 1))])
    affine, *_ = np.linalg.lstsq(X, dst, rcond=None)  # 3x2: dst = X @ affine

    resid = np.linalg.norm(X @ affine - dst, axis=1)
    if np.median(resid) > 2.0:
        raise RuntimeError(f"Diode-photo registration residual too high (median={np.median(resid):.2f}px) "
                            "-- inspect the photo/rotation before trusting the calibration.")

    m = affine[:2, :].T
    t = affine[2, :]
    m_inv = np.linalg.inv(m)
    c_inv = -m_inv @ t
    pil_coeffs = (m_inv[0, 0], m_inv[0, 1], c_inv[0], m_inv[1, 0], m_inv[1, 1], c_inv[1])

    height, width = pixel_labels.shape
    registered = photo.transform((width, height), Image.AFFINE, pil_coeffs, resample=Image.BILINEAR)
    return np.array(registered).astype(float)


def _hexagon_divider_angle(pixel_labels, registered_photo_lum, label, erode_iters=6,
                            dark_percentile=15, min_points=15):
    """
    PCA angle (degrees, mod 180) of the divider line's darkest pixels within one hexagon.
    Eroding the mask first drops the ring of pixels along the hexagon's own dark border, so
    what's left is dominated by the divider line rather than border bleed-through.
    """
    mask = pixel_labels == label
    eroded = ndimage.binary_erosion(mask, iterations=erode_iters)
    ys, xs = np.nonzero(eroded)
    if len(ys) == 0:
        return None, 0.0
    vals = registered_photo_lum[ys, xs]
    threshold = np.percentile(vals, dark_percentile)
    selected = vals <= threshold
    if selected.sum() < min_points:
        return None, 0.0
    coords = np.column_stack([xs[selected], ys[selected]]).astype(float)
    coords -= coords.mean(axis=0)
    _, singular_values, vt = np.linalg.svd(coords, full_matrices=False)
    angle = np.degrees(np.arctan2(vt[0, 1], vt[0, 0])) % 180
    elongation = singular_values[0] / (singular_values[1] + 1e-6)
    return angle, elongation


def calibrate_split_angles(pixel_labels, photo_path=DIODE_PHOTO_FILE, confidence_elongation=2.8):
    """
    Measures the two mirror-image divider angles and confirms they alternate strictly by row.
    Returns (angle_even_row, angle_odd_row, row_pitch, y0) for use by build_diode_side_labels().
    """
    registered_lum = _register_diode_photo(pixel_labels, photo_path).sum(axis=-1)
    labels, centroids = _hexagon_centroids(pixel_labels)

    y_values = np.sort(np.unique(np.round(centroids[:, 0], 1)))
    row_gaps = np.diff(y_values)
    row_pitch = np.median(row_gaps[row_gaps > 30])
    y0 = y_values[0]
    row_id = np.round((centroids[:, 0] - y0) / row_pitch).astype(int)

    angles = np.full(len(labels), np.nan)
    elongations = np.zeros(len(labels))
    for i, label in enumerate(labels):
        angle, elongation = _hexagon_divider_angle(pixel_labels, registered_lum, label)
        if angle is not None:
            angles[i] = angle
            elongations[i] = elongation

    confident = elongations > confidence_elongation
    even_angles = angles[confident & (row_id % 2 == 0)]
    odd_angles = angles[confident & (row_id % 2 == 1)]
    if len(even_angles) < 20 or len(odd_angles) < 20 or np.std(even_angles) > 3 or np.std(odd_angles) > 3:
        raise RuntimeError("Divider-angle calibration is not clean enough to trust -- inspect the "
                            "registered photo before proceeding.")

    return float(np.median(even_angles)), float(np.median(odd_angles)), float(row_pitch), float(y0)


def build_diode_side_labels(pixel_labels, angle_even_row, angle_odd_row, row_pitch, y0):
    """
    Splits every hexagon's pixel mask in half along a line through its centroid, using
    angle_even_row / angle_odd_row depending on that hexagon's row parity.
    """
    labels, centroids = _hexagon_centroids(pixel_labels)
    row_id = np.round((centroids[:, 0] - y0) / row_pitch).astype(int)

    diode_side = np.zeros_like(pixel_labels, dtype=np.uint8)
    for label, (cy, cx), row in zip(labels, centroids, row_id):
        theta = np.radians(angle_even_row if row % 2 == 0 else angle_odd_row)
        normal_x, normal_y = -np.sin(theta), np.cos(theta)

        mask = pixel_labels == label
        ys, xs = np.nonzero(mask)
        side = (xs - cx) * normal_x + (ys - cy) * normal_y > 0
        diode_side[ys[side], xs[side]] = 2
        diode_side[ys[~side], xs[~side]] = 1

    return diode_side


def main():
    with open(PIXEL_LABEL_FILE, "rb") as f:
        pixel_labels = pickle.load(f)

    angle_even_row, angle_odd_row, row_pitch, y0 = calibrate_split_angles(pixel_labels)
    print(f"Calibrated split angles: even rows={angle_even_row:.2f} deg, "
          f"odd rows={angle_odd_row:.2f} deg (row_pitch={row_pitch:.2f}px, y0={y0:.2f}px)")

    diode_side = build_diode_side_labels(pixel_labels, angle_even_row, angle_odd_row, row_pitch, y0)

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(diode_side, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
