
#  hast-ucc: a hasty implementation for spin-unrestricted coupled cluster
#  Copyright (C) 2025 Huanchen Zhai <hczhai.ok@gmail.com>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program. If not, see <https://www.gnu.org/licenses/>.
#
#

def gt_initialize(inits, ctr_idxs):
    ill, irr = [set(x[i] for x in ctr_idxs) for i in [0, 1]]
    nlr, nll, nrr = [sum(inits[x] for x in i) for i in [ill.intersection(irr), ill, irr]]
    if not ((nrr >= nll and nrr <= nll + nlr) or (nll >= nrr and nll <= nrr + nlr)):
        return []
    ts, nctr = list(inits), sum(inits) // 2
    def seq_search(ictr, pctr, r=[], eq=[0] * nctr):
        if ictr == nctr:
            r.append(tuple(eq))
        else:
            for ip, (i, j) in enumerate(ctr_idxs[pctr:]):
                if ts[i] > (i == j) and ts[j] > (i == j):
                    ts[i], ts[j] = ts[i] - 1 - (i == j), ts[j] - 1 - (i == j)
                    eq[ictr] = (1 << i) | (1 << j)
                    seq_search(ictr + 1, ip + pctr, r, eq)
                    ts[i], ts[j] = ts[i] + 1 + (i == j), ts[j] + 1 + (i == j)
        return r
    return seq_search(0, 0)

def is_conn(eq, pts, inits=None):
    inits = inits if inits is not None else [pts[0]]
    cns, pts = [1 << x for x in inits], set([1 << x for x in pts if x not in inits])
    while len(cns) != 0:
        q = cns.pop()
        for x in eq:
            if x & q and (x ^ q) in pts:
                pts.remove(x ^ q)
                cns.append(x ^ q)
    return len(pts) == 0

def gt_canonicalize(eq, nts, names, deq=None, deq_base=2, nx=2):
    from functools import reduce
    if deq is None:
        msort = lambda ixs: sorted([sum(((x >> q) & 1) << p for p, q in enumerate(ixs[::-1])) for x in eq])
    else:
        msort = lambda ixs: sorted([sum(((x >> (q * deq_base * nx + r)) & 1) << (p * deq_base * nx + r)
            for p, q in enumerate(ixs[::-1]) for r in range(deq_base * nx)) for x in deq])
    msort_nm = lambda ixs: [names[q] for q in ixs[::-1]]
    conn_mat = [reduce(lambda a, b: a | b, [x ^ (1 << i) for x in eq if x & (1 << i)], 0) for i in range(nts)]
    gix = [((), [len([j for j in range(i + 1, nts) if conn_mat[i] & (1 << j)]) for i in range(nts)])]
    for _ in range(nts):
        gmat, gnm, pix, gix = None, None, gix, []
        for j, jxs, uroots in [(j, ixs + (j, ), rs) for ixs, rs in pix for j, jr in enumerate(rs) if jr == 0]:
            rmat, rnm = msort(jxs), msort_nm(jxs)
            if gmat is None or (rnm, rmat) < (gnm, gmat):
                gmat, gnm, gix = rmat, rnm, []
            if (rnm, rmat) <= (gnm, gmat):
                jroots = [u - (i == j) for i, u in enumerate(uroots)]
                for p in range(j):
                    jroots[p] -= (conn_mat[j] >> p) & 1
                gix.append((jxs, jroots))
    return tuple(msort(gix[0][0])), tuple(msort_nm(gix[0][0]))

def gt_notes(is_mr, is_hbar):
    mt = {'': [(0, 1, 0), (0, 1, 1), (1, 0, 2)] if is_mr else [(0, 1, 0), (1, 0, 1)]}
    if is_mr:
        mt['TT'] = mt['TH'] = mt['TG'] = mt['AA'] = mt['HA'] = mt['GA'] = mt['TA'] = mt['TX'] = mt['XA'] = [(0, 1, 1)]
        mt['T#'] = mt['R#'] = mt['A#'] = mt['L#'] = mt['H#'] = mt['G#'] = mt['X#'] = [(0, 1, 1), (1, 0, 1)]
        mt['T^'] = mt['R^'] = mt['A^'] = mt['L^'] = mt['H^'] = mt['G^'] = mt['X^'] = [(0, 1, 1), (1, 0, 1)]
    mt[['Z#', 'X#'][is_hbar]] = mt[['Z^', 'X^'][is_hbar]] = []
    mt[['ZA', 'XA'][is_hbar]] = [(b, a, c) for a, b, c in mt['']] + ([(0, 1, 1)] if is_mr else [])
    mt[['ZH', 'XH'][is_hbar]] = mt['ZX'] = mt[''] + [(b, a, c) for a, b, c in mt['']]
    mt[['ZT', 'XT'][is_hbar]] = mt[''] + [(b, a, c) for a, b, c in [(0, 1, 1)] if is_mr]
    mt[['ZR', 'XR'][is_hbar]] = mt[''] + [(b, a, c) for a, b, c in [(0, 1, 1)] if is_mr]
    return mt

def get_letter_index(ref, xptr, idxd, ix, idx):
    if idx in idxd[ix]:
        return idxd[ix][idx]
    else:
        rptr = (xptr[ix] - 1 + len(ref)) % len(ref)
        while rptr != xptr[ix] and ref[xptr[ix]] == ' ':
            xptr[ix] = (xptr[ix] + 1) % len(ref)
        if rptr == xptr[ix]:
            raise RuntimeError('The alphabet is exhausted!')
        else:
            idxd[ix][idx], ref[xptr[ix]] = ref[xptr[ix]], ' '
            return idxd[ix][idx]

def get_perm_sign(perm):
    p, sign = list(perm), 1
    for j in (j for i in range(len(p) - 1) for j in range(i + 1)[::-1] if p[j] > p[j + 1]):
        sign, p[j], p[j + 1] = -sign, p[j + 1], p[j]
    return sign

def gt_idx_dsu(conn):
    import itertools
    rootx = {k : k for cn in conn for k in cn}
    def findx(x):
        if rootx[x] != x:
            rootx[x] = findx(rootx[x])
        return rootx[x]
    def unionx(x, y):
        x, y = findx(x), findx(y)
        if x != y:
            rootx[x] = y
    for ca, cb in conn:
        unionx(ca, cb)
    return [tuple(x) for _, x in itertools.groupby(sorted(rootx, key=findx), key=findx)]

def gt_npdm_symmetrize(r):
    return (r + r.transpose(*range(r.ndim // 2, r.ndim), *range(0, r.ndim // 2))) / 2
