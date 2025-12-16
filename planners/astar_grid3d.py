import heapq
import numpy as np

def astar_3d(occ: np.ndarray, start, goal, diag=True):
    """
    3D grid A* on occupancy grid.
    occ: (Nx,Ny,Nz) bool, True=obstacle
    start/goal: tuple (ix,iy,iz)
    diag: allow 26-neighborhood if True else 6-neighborhood
    Return: list of grid indices [(ix,iy,iz), ...] or None
    """
    Nx, Ny, Nz = occ.shape
    sx, sy, sz = start
    gx, gy, gz = goal

    def inb(x,y,z):
        return (0 <= x < Nx) and (0 <= y < Ny) and (0 <= z < Nz)

    if not inb(*start) or not inb(*goal):
        return None
    if occ[start] or occ[goal]:
        return None

    if diag:
        nbrs = [(dx,dy,dz) for dx in (-1,0,1)
                        for dy in (-1,0,1)
                        for dz in (-1,0,1)
                        if not (dx==0 and dy==0 and dz==0)]
    else:
        nbrs = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

    def h(x,y,z):
        # Euclidean heuristic
        return np.sqrt((x-gx)**2 + (y-gy)**2 + (z-gz)**2)

    g_cost = {start: 0.0}
    parent = {start: None}

    pq = []
    heapq.heappush(pq, (h(*start), start))

    while pq:
        f, cur = heapq.heappop(pq)
        if cur == goal:
            # reconstruct
            path = []
            p = cur
            while p is not None:
                path.append(p)
                p = parent[p]
            path.reverse()
            return path

        cx, cy, cz = cur
        gc = g_cost[cur]

        for dx,dy,dz in nbrs:
            nx, ny, nz = cx+dx, cy+dy, cz+dz
            if not inb(nx,ny,nz):
                continue
            if occ[nx,ny,nz]:
                continue

            step = np.sqrt(dx*dx + dy*dy + dz*dz)
            ng = gc + step
            nxt = (nx,ny,nz)

            if (nxt not in g_cost) or (ng < g_cost[nxt]):
                g_cost[nxt] = ng
                parent[nxt] = cur
                heapq.heappush(pq, (ng + h(nx,ny,nz), nxt))

    return None

def grid_path_to_world(path_idx, origin, res):
    """
    Convert grid indices to world coordinates (center of cell).
    origin: (x0,y0,z0)
    res: cell size (m)
    """
    origin = np.asarray(origin, float).reshape(3)
    pts = []
    for (ix,iy,iz) in path_idx:
        pts.append(origin + res*(np.array([ix,iy,iz], float) + 0.5))
    return np.vstack(pts)

def shortcut_smooth(points, occ=None, origin=None, res=None, n_iter=200, seed=0):
    """
    Simple shortcut smoothing on waypoint polyline.
    If occ/origin/res provided, will check line collision using sampling.
    """
    rng = np.random.default_rng(seed)
    pts = points.copy()
    if pts.shape[0] <= 2:
        return pts

    def collision_free(p, q):
        if occ is None:
            return True
        # sample along segment in world, convert to grid and check
        seg = q - p
        L = np.linalg.norm(seg)
        if L < 1e-9:
            return True
        n = int(np.ceil(L / (0.5*res)))  # sample at ~half cell
        for k in range(n+1):
            a = k / max(1,n)
            x = p + a*seg
            ijk = np.floor((x - origin) / res).astype(int)
            ix,iy,iz = ijk.tolist()
            if ix<0 or iy<0 or iz<0 or ix>=occ.shape[0] or iy>=occ.shape[1] or iz>=occ.shape[2]:
                return False
            if occ[ix,iy,iz]:
                return False
        return True

    for _ in range(n_iter):
        i = rng.integers(0, pts.shape[0]-1)
        j = rng.integers(0, pts.shape[0]-1)
        if abs(i-j) < 2:
            continue
        a, b = min(i,j), max(i,j)
        if collision_free(pts[a], pts[b]):
            pts = np.vstack([pts[:a+1], pts[b:]])
            if pts.shape[0] <= 2:
                break
    return pts
