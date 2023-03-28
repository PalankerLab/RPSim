import numpy as np

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