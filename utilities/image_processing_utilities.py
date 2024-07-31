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
    Params:
        img_in (Numpy.array (M,N,3)): A 3 channels numpy array containing the subframe to project
        label (Numpy.array (M,N)): A 1 channel numpy array containing the photodiode location of each pixel
                                    each pixel's photodiode is encoded by the same integer, the pixel's label
    Return
        A 1-D array, each entry contains the light on a single pixel
    """
    # convert to grayscale
    img = img_in[:, :, 0] * 0.2989 + img_in[:, :, 1] * 0.5870 + img_in[:, :, 2] * 0.1141
    assert (img.shape == label.shape), f"Dimension inconsistent between the input image {img.shape} and the label map {label.shape}."

    N_pixels = label.max()
    light_on_pixels = np.zeros(N_pixels)
    for kk in range(N_pixels):
        # label is the segmentation bitmap of all pixels
        # (label == kk+1) is a mask selecting the photodiode of single pixel 
        # The mask is applied to the incoming image and the brightness is averaged
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

# TODO: n_components into user config
def Rmat_simp(Rmat, ratio, imag_basis, Gs=0):
    N = Rmat.shape[0]
    G_comp = {}
    # Invert the resistance matrix to obtain the conductance matrix
    Gmat = 1 / Rmat
    # Calculate the Laplacian-like matrix G
    G = np.diagflat(np.sum(Gmat, axis=0)) - np.tril(Gmat, -1) - np.triu(Gmat, 1)
    # Sort absolute values of G and determine the threshold for sparsification
    G_sort = np.sort(np.abs(G), axis=None)
    threshold = G_sort[-int(N**2 * ratio)]
    discard_idx = np.array(np.abs(G) < threshold)
    # Sparsify G and create the sparse resistance matrix
    S = np.array(G)
    S[discard_idx] = 0
    Smat = np.diagflat(np.sum(S, axis=0)) - np.tril(S, -1) - np.triu(S, 1)
    Smat[discard_idx] = np.nan
    Rmat_sparse = 1 / Smat
    # Compute the error matrix E
    E = G - S
    # Eigenvalue decomposition of E
    (w, V) = np.linalg.eigh(E)
    # Sort the eigenvalues by their absolute values
    sorted_indices = np.argsort(np.abs(w))
    # Use the first 50 principal components for general compensation
    n_principal_components = 150 # The number or desired principal components for general compensation
    num_components = min(n_principal_components, len(w))  # Ensure we don't exceed the number of available components
    error_plot = np.zeros(n_principal_components+1)
    error_plot[0] = np.linalg.norm(E)
    for i in range(num_components):
        w_idx = sorted_indices[-(i + 1)]
        E = E - w[w_idx] * np.outer(V[:, w_idx], V[:, w_idx])
        error_plot[i+1] = np.linalg.norm(E)
    # Image specific compensation using expected voltage/current
    v_basis = np.linalg.solve(G + np.eye(N) * Gs, imag_basis)
    (v_basis_om, _) = np.linalg.qr(v_basis)
    i_basis = E.dot(v_basis_om)
    basis_idx = np.linalg.norm(i_basis, axis=0) > np.abs(w[sorted_indices[-1]]) * 1E-3
    #plt10.figure()
    #plt10.plot(error_plot)
    #plt10.savefig('error_M.png')
    # Construct compensation bases using principal components and specific comp/
    included_gen_comp_indices = sorted_indices[-num_components:]
    G_comp['v_basis'] = np.concatenate([V[:, idx].flatten() for idx in included_gen_comp_indices] + [v_basis_om[:, basis_idx].flatten()])
    G_comp['v_basis'] = G_comp['v_basis'].reshape((-1, N)).T
    G_comp['i_basis'] = np.concatenate([V[:, idx].flatten() * w[idx] for idx in included_gen_comp_indices] + [i_basis[:, basis_idx].flatten()])
    G_comp['i_basis'] = G_comp['i_basis'].reshape((-1, N)).T
    # included_gen_comp_indices = sorted_indices[-num_components:]
    # G_comp['v_basis'] = np.concatenate([V[:, idx].flatten() for idx in included_gen_comp_indices])
    # G_comp['v_basis'] = G_comp['v_basis'].reshape((-1, N)).T
    # G_comp['i_basis'] = np.concatenate([V[:, idx].flatten() * w[idx] for idx in included_gen_comp_indices])
    # G_comp['i_basis'] = G_comp['i_basis'].reshape((-1, N)).T
    return Rmat_sparse, G_comp