import numpy as np
import numpy.linalg as nla

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
class Green:
    def __init__(self, data) -> None:
        if type(data) is tuple:
            x, v = data
        else:
            xv = np.array(np.loadtxt(data, delimiter=','))
            x = xv[:,0]
            v = xv[:,1]

        self.x = np.array(x)
        idx = self.x.argsort()
        self.x = self.x[idx]
        self.v = np.array(v)[idx]

        x10 = self.x[self.x<10]
        y10 = 1/self.v[self.x<10]
        self.asymt_k = (x10@x10)/(x10@y10)
    
    def inquire(self, x_q):
        v_q = np.interp(x_q, self.x, self.v, left=np.nan)
        small_idx = np.logical_and(np.isnan(v_q), x_q!=0)
        v_q[small_idx] = self.asymt_k / x_q[small_idx]
        v_q[x_q==0] = 0

        return v_q

class Point:
    def __init__(self, xy) -> None:
        assert len(xy) == 2
        self.p = np.array(xy)
    
    def inside(self, t):
        for kk in range(3):
            v = getattr(t, f'v{kk+1}')
            if nla.norm(v - self.p) < t.scale * 1E-6:
                return kk+1 # at one of the vertices
        for kk in range(3):
            v = getattr(t, f'v{kk+1}')
            n = getattr(t, f'n{kk+1}')
            if (self.p - v)@n < 0:
                return False
        return True
    
class Triangle:
    mesh_N = 100
    @staticmethod
    def mesh_gen(N=mesh_N):
        x = np.arange(0, 1+1/N, 1/N)
        xv, yv = np.meshgrid(x, x)
        xc = np.concatenate([xv + (kk+1)/3/N for kk in range(2)])
        yc = np.concatenate([yv + (kk+1)/3/N for kk in range(2)])

        cnstrt = xv + yv
        xv = xv[cnstrt<=1]
        yv = yv[cnstrt<=1]
        mesh_v = np.array([xv, yv]).T
        
        cnstrt = xc + yc
        xc = xc[cnstrt<1]
        yc = yc[cnstrt<1]
        mesh_c = np.array([xc, yc]).T

        return mesh_v, mesh_c
    mesh_norm_v, mesh_norm_c = mesh_gen.__func__()

    @staticmethod
    def mesh_v_weight(mesh=mesh_norm_v):
        wgt = np.ones(mesh.shape[0])/2
        wgt[mesh[:,0]==0] = 1/4
        wgt[mesh[:,1]==0] = 1/4
        wgt[np.sum(mesh, axis=1)==1] = 1/4
        wgt[np.sum(mesh, axis=1)==0] = 1/12
        wgt[np.logical_and(mesh[:,0]==0, mesh[:,1]==1)] = 1/12
        wgt[np.logical_and(mesh[:,0]==1, mesh[:,1]==0)] = 1/12

        return wgt
        
    mesh_wgt_v = mesh_v_weight.__func__()
    mesh_wgt_c = 3/4 * np.ones(mesh_norm_c.shape[0])
    
    def __init__(self, coords) -> None:
        self.coords = np.array(coords)
        assert self.coords.shape == (3, 2)

        self.v1 = np.array(self.coords[0,:])
        self.v2 = np.array(self.coords[1,:])
        self.v3 = np.array(self.coords[2,:])
        self.center = np.mean(coords, axis=0)
        self.e1 = self.v2 - self.v1
        self.e2 = self.v3 - self.v2
        self.e3 = self.v1 - self.v3
        self.n1 = self.e2 - (self.e2 @ self.e1) / nla.norm(self.e1)**2 * self.e1 #normal to e1
        self.n2 = self.e3 - (self.e3 @ self.e2) / nla.norm(self.e2)**2 * self.e2 #normal to e2
        self.n3 = self.e1 - (self.e1 @ self.e3) / nla.norm(self.e3)**2 * self.e3 #normal to e3
        self.scale = np.max(nla.norm([self.e1, self.e2, self.e3], axis=1))

        self.mesh_c = Triangle.mesh_norm_c @ np.array([self.e1, -self.e3]) + self.v1
        self.mesh_v = Triangle.mesh_norm_v @ np.array([self.e1, -self.e3]) + self.v1
        self.S = np.abs(np.cross(self.e1, self.e2)) / 2   #area of eahc mesh triangle
        self.dS = self.S / Triangle.mesh_N**2   #area of eahc mesh triangle

    def green_int(self, target:Point, G:Green):
        topo = target.inside(self)
        r_c, r_v = self.dist_on(target, topo) if topo else self.dist_off(target)
        v_c = G.inquire(r_c) * Triangle.mesh_wgt_c
        v_v = G.inquire(r_v) * Triangle.mesh_wgt_v
        v = ( np.sum(v_c) + np.sum(v_v) ) / Triangle.mesh_N**2
        return v

    def dist_off(self, target:Point):  # used when the target point is outside the triangle
        r_c = nla.norm(target.p - self.mesh_c, axis=1)
        r_v = nla.norm(target.p - self.mesh_v, axis=1)
        return r_c, r_v

    def dist_on(self, target:Point, topo):  # used when the target point is one of the vertices
        if type(topo) is int:
            pt = getattr(self, f'v{topo}')
            r_c = nla.norm(pt - self.mesh_c, axis=1)
            r_v = nla.norm(pt - self.mesh_v, axis=1)

            idx = np.argmin(nla.norm(self.mesh_v - pt, axis=1))
            idx_v = nla.norm(Triangle.mesh_norm_v - Triangle.mesh_norm_v[idx], axis=1) < 1.01/Triangle.mesh_N
            r_v[idx_v] = r_v[idx_v] * 1.5
            idx_c = np.argmin(nla.norm(self.mesh_c - pt, axis=1))
            r_c[idx_c] = 0

            e1 = getattr(self, f'e{topo}')
            e2 = getattr(self, f'e{topo-1 if topo>1 else 3}')
            theta = np.arccos((e1@e2)/(nla.norm(e1)*nla.norm(e2)))
            R_eq = np.sqrt(self.dS/theta)
            r_v[idx] = 1/(12*R_eq*theta)
        else:
            idx = np.argmin(nla.norm(self.mesh_c - target.p, axis=1))
            pt = self.mesh_c[idx, :]
            r_c = nla.norm(pt - self.mesh_c, axis=1)
            r_v = nla.norm(pt - self.mesh_v, axis=1)

            R_eq = np.sqrt(0.75*self.dS/np.pi)
            r_c[idx] = 1/(2*np.pi*R_eq)
        return r_c, r_v

    def tri_mutual(self, t, G:Green):
        r_m = self.green_int(Point(t.center), G) * 0.75
        for kk in range(3):
            p = Point(getattr(t, f'v{kk+1}'))
            r_m += self.green_int(p, G) / 12
        return r_m



