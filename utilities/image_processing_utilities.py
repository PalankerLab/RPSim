import numpy as np
from PIL import Image as im


def red_corners(img_in, n_target=1500):
    """
    This function finds the proper zoom range labeled by the red box to convert to projection pattern
    """
    M, N = img_in.shape[0:2]
    img = img_in.astype(float)
    img_red = img[:, :, 0] > img[:, :, 1:].sum(axis=2)

    if img_red.any():
        # build global coordinate
        x = np.array([kk for kk in range(N)])
        y = np.array([kk for kk in range(M)])
        xx, yy = np.meshgrid(x, y)
        dist_mat = (xx+yy).astype(float)
        dist_mat[~img_red] = np.nan

        # find the top-left and bottom-right red corners
        corner_idx = [np.nanargmin(dist_mat), np.nanargmax(dist_mat)]
        x_min, x_max = xx.flatten()[corner_idx]
        y_min, y_max = yy.flatten()[corner_idx]

        img = img[y_min:y_max+1, x_min:x_max+1]

    size_target = (n_target, n_target)
    # stretch the frame properly
    if not img.shape == size_target:
        img = im.fromarray(np.uint8(img.round()))
        img = img.resize(size_target)
        img = np.array(img).astype(float)

    return img


def int_sq(x, v_in):
    """
    This function integrates in the polar coordinate.
    """
    x = x.flatten()**2
    v = np.sum(x * v_in.flatten()) / x.sum()
    return v


def img2pixel(img_in, label):
    """
    This function converts the projection image to the relative light flux on each pixel
    """
    # convert to grayscale
    img = img_in[:, :, 0] * 0.2989 + img_in[:, :, 1] * 0.5870 + img_in[:, :, 2] * 0.1141
    assert (img.shape == label.shape), "Dimension inconsistent between the input image and the label map."

    N_pixels = label.max()
    light_on_pixels = np.zeros(N_pixels)
    for kk in range(N_pixels):
        # label is the segmentation bitmap of all pixels
        light_on_pixels[kk] = np.mean(img[label == kk+1])

    return np.round(light_on_pixels/255, decimals=6).tolist()


def is_edge(px_pos, px_size, neighbors=6):
    """
    This function determines if a bipolar pixel is at the edge of an array and hence the return area is different
    """
    x = px_pos[:, 0]
    y = px_pos[:, 1]
    xx = x - x[:, np.newaxis]
    yy = y - y[:, np.newaxis]

    dist = np.sqrt(xx**2 + yy**2) - px_size
    is_neighbor = np.abs(dist) < 0.1
    neighbor_num = np.sum(is_neighbor, axis=0)

    return neighbor_num < neighbors

def Rmat_simp(Rmat, ratio, imag_basis, Gs=0):
    N = Rmat.shape[0]
    G_comp = {}

    Gmat = 1/Rmat
    G = np.diagflat(np.sum(Gmat, axis=0)) - np.tril(Gmat, -1) - np.triu(Gmat, 1)

    G_sort = np.sort(np.abs(G), axis=None)
    threshold = G_sort[ -int(N**2 *ratio) ]
    discard_idx = np.array(np.abs(G) < threshold)
    
    S = np.array(G)
    S[discard_idx] = 0
    Smat = np.diagflat(np.sum(S, axis=0)) - np.tril(S, -1) - np.triu(S, 1)
    Smat[discard_idx] = np.nan
    Rmat_sparse = 1/Smat

    E = G - S
    (w, V) = np.linalg.eigh(E)
    w_idx = np.argsort(np.abs(w))[-1]
    
    E = E - w[w_idx] * np.outer(V[:, w_idx], V[:, w_idx])
    v_basis = np.linalg.solve(G + np.eye(N)*Gs, imag_basis)
    (v_basis_om, _) = np.linalg.qr(v_basis)
    i_basis = E.dot(v_basis_om)
    basis_idx = np.linalg.norm(i_basis, axis=0) > np.abs(w[w_idx])*1E-3
    
    G_comp['v_basis'] = np.concatenate((V[:, w_idx].flatten(), v_basis_om[:, basis_idx].flatten()))
    G_comp['v_basis'] = G_comp['v_basis'].reshape((-1, N)).T
    G_comp['i_basis'] = np.concatenate((V[:, w_idx].flatten()*w[w_idx], i_basis[:, basis_idx].flatten()))
    G_comp['i_basis'] = G_comp['i_basis'].reshape((-1, N)).T

    return Rmat_sparse, G_comp
