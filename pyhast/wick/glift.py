
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

def gt_pre_complexity(eq, nts, names, deq_base, nx, shapes):
    ctr_idx = [0] * nx
    tidxs = [[0 for _ in range(nts)] for _ in range(nx)]
    for x in eq:
        zc, zd = [(n, i % nx) for n in range(nts) for i in range(deq_base * nx) if x & (1 << (n * deq_base * nx + i))]
        tidxs[zc[1]][zc[0]] |= 1 << ctr_idx[zc[1]]
        tidxs[zd[1]][zd[0]] |= 1 << ctr_idx[zd[1]]
        ctr_idx[zc[1]], ctr_idx[zd[1]] = ctr_idx[zc[1]] + 1, ctr_idx[zd[1]] + 1
    for ix in range(nx):
        tidxs[ix] = [x for x, nm in zip(tidxs[ix], names) if nm != 'X']
    if '#' not in names and '^' not in names: # EI or EIA
        return [*tidxs[2:], *[tidxs[ii] for ii in sorted([0, 1], key=lambda i: -shapes['EI'[i]] if shapes is not None else 0)]]
    else:
        return [*tidxs[3:], *[tidxs[ii] for ii in sorted([0, 2, 1], key=lambda i: -shapes['EAI'[i]] if shapes is not None else 0)]]

def gt_complexity(eq, nts, names, deq_base, nx, shapes, use_hint=True):
    from functools import reduce
    tidxs = gt_pre_complexity(eq, nts, names, deq_base, nx, shapes)
    nn = len(tidxs[0])
    pop = [0] * max(1 << nn, reduce(lambda x, y: x | y, reduce(lambda x, y: x + y, tidxs)) + 1)
    for i in range(len(pop)):
        pop[i] = pop[i >> 1] + (i % 2 != 0)
    r = [None] * (1 << nn)
    i_used = [False] * nn
    if use_hint and '#' not in names and '^' not in names and 'G' not in names and nx == 2:
        for i in range(nn):
            for j in range(i + 1, nn):
                if not i_used[i] and not i_used[j]:
                    pav, pao = pop[tidxs[0][i]], pop[tidxs[1][i]]
                    pbv, pbo = pop[tidxs[0][j]], pop[tidxs[1][j]]
                    pko, pkv = pop[tidxs[0][i] & tidxs[0][j]], pop[tidxs[1][i] & tidxs[1][j]]
                    pa, pb, pk = pav + pao, pbv + pbo, pkv + pko
                    pc = pa + pb - 2 * pk
                    pabv = tidxs[0][i] ^ tidxs[0][j]
                    pabo = tidxs[1][i] ^ tidxs[1][j]
                    if nn == 3 and pc == 2 and (pa == 4 or pb == 4):
                        i_used[i] = i_used[j] = True
                        r[(1 << i) | (1 << j)] = ((pabv, pabo), [(pav + pbv - pkv, pao + pbo - pko)], (i, j))
                    elif pc == 4 and pa == 2 and pb == 2:
                        i_used[i] = i_used[j] = True
                        r[(1 << i) | (1 << j)] = ((pabv, pabo), [(pav + pbv - pkv, pao + pbo - pko)], (i, j))
    for i in range(nn):
        if not i_used[i]:
            r[1 << i] = (tuple(tidxs[ix][i] for ix in range(nx)), [], i)
    for i in range(1, 1 << nn):
        mx = [k for k in range(nn) if i & (1 << k)]
        # mx is list of included tensors
        mord, midx, mxy = None, None, None
        for j in range(1, (1 << pop[i]) - 1):
            # j is pattern of left in i
            x = j
            for ki, kj in list(enumerate(mx))[::-1]:
                if ki != kj and (x & (1 << ki)):
                    x = x ^ (1 << ki) ^ (1 << kj)
            y = i ^ x
            if r[x] is None or r[y] is None:
                continue
            # x is left part of i, y is right part of i
            xord = r[x][1] + r[y][1] + [tuple(pop[r[x][0][k]] + pop[r[y][0][k]] - (pop[r[x][0][k] & r[y][0][k]]) for k in range(nx))]
            xord = sorted(xord, key=lambda x: [sum(x), *x])[::-1]
            xidx = tuple(r[x][0][k] ^ r[y][0][k] for k in range(nx))
            if mord is None or [(sum(x), *x) for x in xord] < [(sum(x), *x) for x in mord]:
                mord, midx, mxy = xord, xidx, (r[x][2], r[y][2])
        if mord is not None:
            r[i] = (midx, mord, mxy)
    return r[(1 << nn) - 1][1:]

def assoc_flatten(assoc):
    if isinstance(assoc, tuple):
        return assoc_flatten(assoc[0]) + assoc_flatten(assoc[1])
    else:
        return [assoc]

def assoc_reorder(assoc, perm):
    if isinstance(assoc, tuple):
        return (assoc_reorder(assoc[0], perm), assoc_reorder(assoc[1], perm))
    else:
        return perm.index(assoc)

def assoc_in(x, xs):
    if isinstance(xs, tuple):
        return assoc_in(x, xs[0]) or assoc_in(x, xs[1])
    else:
        return x == xs

def gt_reorder(deq, nts, names, assoc, deq_base, nx):
    idd = names.index('X')
    perm = [x for x in list(range(nts)) if x != idd]
    fperm = assoc_flatten(assoc)
    perm = [perm[x] for x in fperm] + [idd]
    inv_perm = [perm.index(x) for x in range(nts)]
    neq = []
    for x in deq:
        (zc, zci), (zd, zdi) = [(n, i) for n in range(nts) for i in range(deq_base * nx) if x & (1 << (n * deq_base * nx + i))]
        neq.append((1 << (inv_perm[zc] * deq_base * nx + zci)) | (1 << (inv_perm[zd] * deq_base * nx + zdi)))
    deq, names = tuple(neq), tuple(names[p] for p in perm)
    assoc = assoc_reorder(assoc, fperm)
    return deq, names, assoc

def gt_multi_extract(eq, nts, names, deq_base, nx, assoc):
    if not isinstance(assoc, tuple):
        return [] if nts != 2 else [gt_extract(eq, nts, names, deq_base, nx, assoc)]
    else:
        a = gt_multi_extract(eq, nts, names, deq_base, nx, assoc[0])
        b = gt_multi_extract(eq, nts, names, deq_base, nx, assoc[1])
        return a + b + [gt_extract(eq, nts, names, deq_base, nx, assoc)]

def gt_extract(eq, nts, names, deq_base, nx, assoc):
    assert names[-1] == 'X'
    neq = []
    if not isinstance(assoc, tuple):
        for x in eq:
            (zc, zci), (zd, zdi) = [(n, i) for n in range(nts) for i in range(deq_base * nx) if x & (1 << (n * deq_base * nx + i))]
            assert zd == nts - 1
            neq.append((1 << (0 * deq_base * nx + zci)) | (1 << (3 * deq_base * nx + zdi)))
        nnm = names
    else:
        a, b = assoc
        for x in eq:
            (zc, zci), (zd, zdi) = [(n, i) for n in range(nts) for i in range(deq_base * nx) if x & (1 << (n * deq_base * nx + i))]
            if assoc_in(zc, a) and assoc_in(zd, b):
                neq.append((1 << (0 * deq_base * nx + zci)) | (1 << (2 * deq_base * nx + zdi)))
            elif assoc_in(zc, b) and assoc_in(zd, a):
                neq.append((1 << (2 * deq_base * nx + zci)) | (1 << (0 * deq_base * nx + zdi)))
            elif not assoc_in(zc, assoc) and not assoc_in(zd, assoc):
                pass
            elif (assoc_in(zc, a) and assoc_in(zd, a)) or (assoc_in(zc, b) and assoc_in(zd, b)):
                pass
            elif assoc_in(zc, a) and zd != nts - 1:
                neq.append((1 << (0 * deq_base * nx + zci)) | (1 << (4 * deq_base * nx + zci)))
            elif assoc_in(zc, b) and zd != nts - 1:
                neq.append((1 << (2 * deq_base * nx + zci)) | (1 << (4 * deq_base * nx + zci)))
            elif assoc_in(zd, a) and zc != nts - 1:
                neq.append((1 << (0 * deq_base * nx + zdi)) | (1 << (4 * deq_base * nx + zdi)))
            elif assoc_in(zd, b) and zc != nts - 1:
                neq.append((1 << (2 * deq_base * nx + zdi)) | (1 << (4 * deq_base * nx + zdi)))
            elif assoc_in(zc, a) and zd == nts - 1 and not isinstance(a, tuple):
                neq.append((1 << (0 * deq_base * nx + zci)) | (1 << (5 * deq_base * nx + zdi)))
            elif assoc_in(zc, a) and zd == nts - 1 and isinstance(a, tuple):
                neq.append((1 << (1 * deq_base * nx + zdi)) | (1 << (5 * deq_base * nx + zdi)))
            elif assoc_in(zc, b) and zd == nts - 1 and not isinstance(b, tuple):
                neq.append((1 << (2 * deq_base * nx + zci)) | (1 << (5 * deq_base * nx + zdi)))
            elif assoc_in(zc, b) and zd == nts - 1 and isinstance(b, tuple):
                neq.append((1 << (3 * deq_base * nx + zdi)) | (1 << (5 * deq_base * nx + zdi)))
            else:
                assert False
        nnm = [a if isinstance(a, tuple) else names[a], b if isinstance(b, tuple) else names[b], 'X']
    return tuple(neq), nnm, assoc

def gt_meq_exchange_ab(meq):
    neq = []
    for a, ia, b, ib, c in meq:
        xx = [1, 0, 2][a], ia, [1, 0, 2][b], ib, c
        if xx[0] > xx[2]:
            xx = xx[2], xx[3], xx[0], xx[1], c
        neq.append(xx)
    return tuple(sorted(neq))

def gt_meq_reorder_using_nm(eq, nts, names, orig_names, ngs, xnm, ix):
    ih = orig_names.index(xnm) if xnm in orig_names else -1
    if ih == -1 or nts == 2 or (isinstance(names[ix], tuple) and ih in assoc_flatten(names[ix])) or names[ix] == xnm:
        return eq, ngs, names
    elif (isinstance(names[1 - ix], tuple) and ih in assoc_flatten(names[1 - ix])) or names[1 - ix] == xnm:
        return gt_meq_exchange_ab(eq), (ngs[1], ngs[0], ngs[2]), (names[1], names[0], names[2])
    else:
        xeq_ab, xeq_ba = eq, gt_meq_exchange_ab(eq)
        if nts == 2 or xeq_ab <= xeq_ba:
            return xeq_ab, ngs, names
        else:
            return xeq_ba, (ngs[1], ngs[0], ngs[2]), (names[1], names[0], names[2])

def gt_lift(ex_eqsss):
    nouts = len(ex_eqsss)
    r = [[] for _ in range(nouts)]
    rev_map = {}
    for ix, ex_eqss in enumerate(ex_eqsss):
        for f, ex_eqs, *symms in [x for x in ex_eqss]:
            tids = {}
            for ixx, (meq, ngs, nm, assoc) in enumerate(ex_eqs):
                tsm = tuple(symms[0][ixx]) if len(symms) != 0 and len(symms[0]) != 0 else None
                tnm = tuple("_%d" % (tids[x][0]) if isinstance(x, tuple) else x for x in nm)
                tng = tuple(tids[x][1] if isinstance(x, tuple) else g for x, g in zip(nm, ngs))
                if ixx == len(ex_eqs) - 1:
                    r[ix].append((f, meq, tng, tnm, tsm))
                else:
                    if (meq, tnm, tsm) not in rev_map:
                        rev_map[(meq, tnm, tsm)] = (len(r) - nouts, ngs[-1])
                        r.append([(1.0, meq, tng, tnm, tsm)])
                    tids[assoc] = rev_map[(meq, tnm, tsm)]
    return nouts, r[nouts:] + r[:nouts]

def gt_meq_is_duplicate(meqa, meqb):
    if len(meqa) != len(meqb):
        return False, 0
    fac = None
    for k in range(len(meqa)):
        xa, xb, xc, xd, xe = meqa[k]
        ya, yb, yc, yd, ye = meqb[k]
        if fac is None:
            fac = xa / ya
        if not (abs(xa - ya * fac) < 1E-10 and xb == yb and xc == yc and xd == yd and xe == ye):
            return False, 0
    return True, fac

def gt_relabel_duplicates(nouts, lf_eqss):
    new_eqss = [sorted([(f, meq, ngs, nm, sm) for f, meq, ngs, nm, sm in eqs], key=lambda x:
        (x[1], x[3], x[4], x[2], abs(x[0]))) for eqs in lf_eqss]
    redt = [(i, 1) for i in range(len(new_eqss))]
    for i in range(len(new_eqss) - nouts):
        for j in range(0, i):
            if redt[j] == (j, 1):
                is_same, fac = gt_meq_is_duplicate(new_eqss[i], new_eqss[j])
                if is_same:
                    redt[i] = (j, fac)
                    break
        if redt[i] != (i, 1):
            for p in range(i + 1, len(new_eqss)):
                for ipx, (f, meq, ngs, nms, sms) in enumerate(new_eqss[p]):
                    new_nms = tuple("%s%d" % (nm[:nm.count('_')], redt[i][0]) if nm in ["_%d" % i, "__%d" % i] else nm for nm in nms)
                    new_f = f
                    for nm in nms:
                        if nm in ["_%d" % i, "__%d" % i]:
                            new_f *= redt[i][1]
                    new_eqss[p][ipx] = (new_f, meq, ngs, new_nms, sms)
    return new_eqss

def gt_meq_conut_desc(meq, nts):
    mts = nts * (nts - 1) // 2
    nc = max([0] + [x[-1] for x in meq]) + 1
    cnts = [[0] * nc for _ in range(mts)]
    outc = tuple(x[-1] for x in sorted(meq, key=lambda x: x[3]) if x[2] == nts - 1)
    for a, _, b, _, c in meq:
        if b == nts - 1:
            cnts[a][c] += 1
        else:
            cnts[mts - 1][c] += 1
    return outc + tuple([tuple(x) for x in cnts])

def gt_meq_desc(meq, symms=None):
    desc = [], []
    for a, ia, b, ib, c in meq:
        if b == 2:
            desc[a].append((0, ia, ib, c))
        else:
            desc[a].append((1, ia, c))
            desc[b].append((1, ib, c))
    if symms is None:
        return tuple(sorted(desc[0])), tuple(sorted(desc[1]))
    else:
        lmap = gt_meq_index_reorder(meq, fidx=0)[1]
        rmap = gt_meq_index_reorder(meq, fidx=1)[1]
        lsymms = tuple(sorted([zm[:2] + tuple(lmap[p][0] for p in zm[2:]) for zm in symms[1]]))
        rsymms = tuple(sorted([zm[:2] + tuple(rmap[p][0] for p in zm[2:]) for zm in symms[0]]))
        return tuple(sorted(desc[0])) + lsymms, tuple(sorted(desc[1])) + rsymms

def gt_remove_unref(nouts, lf_eqss):
    lf_eqss = gt_relabel_duplicates(nouts, lf_eqss)
    ref = [False] * len(lf_eqss)
    queue = [len(lf_eqss) - nouts + x for x in range(nouts)]
    iq = 0
    while iq < len(queue):
        if not ref[queue[iq]]:
            ref[queue[iq]] = True
            for _, _, _, nms, _ in lf_eqss[queue[iq]]:
                for nm in nms:
                    if nm.startswith('_'):
                        queue.append(int(nm[nm.count('_'):]))
        iq += 1
    imap, it = [], 0
    new_eqss = []
    for il in range(len(lf_eqss)):
        imap.append(it)
        if ref[il]:
            new_eqs = []
            for f, meq, ngs, nms, sms in lf_eqss[il]:
                new_nms = ["%s%d" % (nm[:nm.count('_')], imap[int(nm[nm.count('_'):])]) if nm.startswith('_') else nm for nm in nms]
                new_eqs.append((f, meq, ngs, new_nms, sms))
            new_eqss.append(new_eqs)
            it += 1
    return new_eqss

def gt_meq_index_reorder(meq, fidx=0):
    # (0free, 0ctr, 1free, 1ctr)
    nc = max([0] + [x[-1] for x in meq]) + 1
    idxs = [[[], []] for _ in range(nc)]
    for im, (a, ia, b, ib, c) in enumerate(meq):
        if b == 2 and a != fidx:
            idxs[c][0].append([ib, ia, im])
        elif b != 2 and a == fidx:
            idxs[c][1].append([ia, ib, im])
        elif a != 2 and b == fidx:
            idxs[c][1].append([ib, ia, im])
    idxs = [sorted(xx) for x in idxs for xx in x]
    nmeq = [tuple(x) for x in meq]
    smx = sum([len(xx) for xx in idxs])
    xmap = [(-smx, -smx)] * smx
    ip = 0
    for x in idxs:
        for xx in x:
            a, ia, b, ib, c = nmeq[xx[-1]]
            if (b == 2 and a != fidx) or (a != 2 and b == fidx):
                xmap[ia] = (ip, c)
                ia = ip
            elif b != 2 and a == fidx:
                xmap[ib] = (ip, c)
                ib = ip
            nmeq[xx[-1]] = a, ia, b, ib, c
            ip += 1
    return tuple(sorted(nmeq)), xmap

def gt_binary_apply_xmap(meq, xmap):
    nmeq = []
    for a, ia, b, ib, c in meq:
        if a == 2:
            ia = xmap[ia][0]
        elif b == 2:
            ib = xmap[ib][0]
        nmeq.append((a, ia, b, ib, c))
    return tuple(sorted(nmeq))

def gt_unary_apply_xmap(xmap):
    return tuple(sorted([(0, p, 1, q, c) for p, (q, c) in enumerate(xmap)]))

def bipartite_mvc(nl, nr, conn):
    import scipy, numpy as np
    cn = np.array(conn, dtype=int).reshape((-1, 2)).T
    arr = scipy.sparse.coo_matrix((np.ones(len(conn), dtype=int), (cn[0], cn[1])), shape=(nl, nr))
    mr = np.array(scipy.sparse.csgraph.maximum_bipartite_matching(arr))
    vl, vr, gr, xr = np.zeros(nl, dtype=bool), np.zeros(nr, dtype=bool), 0, -1
    vl[mr[mr != -1]] = 1
    while xr != gr:
        vr[cn[1, ~vl[cn[0]]]] = 1
        xr, gr, vl[mr[vr]] = gr, np.sum(vr, dtype=int), 0
    return [int(x) for x in np.mgrid[:nl][vl]], [int(x) for x in np.mgrid[:nr][vr]]

def gt_meq_sum_lift(lf_eqss, nouts):
    pnx = len(lf_eqss)
    nx = pnx - nouts
    queue = [pnx - nouts + x for x in range(nouts)]
    iq = 0
    while iq < len(queue):
        new_lfeqs = []
        eqdt = {}
        for f, meq, ngs, nm, sm in lf_eqss[queue[iq]]:
            if len(nm) == 2:
                new_lfeqs.append((f, meq, ngs, nm, sm))
            else:
                mcnt = gt_meq_conut_desc(meq, len(nm))
                if mcnt not in eqdt:
                    eqdt[mcnt] = []
                eqdt[mcnt].append((f, meq, ngs, nm, sm))
        changed = False
        for mcnt, eqv in eqdt.items():
            nl, nr = 0, 0
            nmml, nmmr = {}, {}
            conn = []
            for f, meq, _, nm, sm in eqv:
                nmds = list(zip(nm, gt_meq_desc(meq, symms=sm)))
                if nmds[0] not in nmml:
                    nmml[nmds[0]] = nl
                    nl += 1
                if nmds[1] not in nmmr:
                    nmmr[nmds[1]] = nr
                    nr += 1
                conn.append((nmml[nmds[0]], nmmr[nmds[1]]))
            vl, vr = bipartite_mvc(nl, nr, conn)
            gl, gr = [[] for _ in vl], [[] for _ in vr]
            mpl = {x:ix for ix, x in enumerate(vl)}
            mpr = {x:ix for ix, x in enumerate(vr)}
            for ix, (_, meq, _, nm, sm) in enumerate(eqv):
                nmds = list(zip(nm, gt_meq_desc(meq, symms=sm)))
                if nmml[nmds[0]] in mpl:
                    gl[mpl[nmml[nmds[0]]]].append(ix)
                else:
                    gr[mpr[nmmr[nmds[1]]]].append(ix)
            for ia, glr in zip([1, 0], [gl, gr]):
                for gs in glr:
                    assert len(gs) > 0
                    f, meq, ngs, nm, sm = eqv[gs[0]]
                    if len(gs) == 1:
                        new_lfeqs.append((f, meq, ngs, nm, sm))
                    else:
                        xmeq, xmap = gt_meq_index_reorder(meq, fidx=1 - ia)
                        xmr_eqs = []
                        for ig in gs:
                            gf, gmeq, gngs, gnm, gsm = eqv[ig]
                            xgmeq, xgmap = gt_meq_index_reorder(gmeq, fidx=1 - ia)
                            assert xgmeq == xmeq
                            if gnm[ia].startswith('_'):
                                (zf, zmeq, zngs, znm, zsm), = lf_eqss[int(gnm[ia][1:])]
                                zsm = zsm[:-1] + (tuple(sorted([zm[:2] + tuple(xgmap[p][0] for p in zm[2:]) for zm in zsm[-1]])), ) if zsm is not None else zsm
                                xmr_eqs.append((gf * zf, gt_binary_apply_xmap(zmeq, xgmap), zngs[:-1] + (None, ), znm, zsm))
                            else:
                                gsm = (gsm[ia], tuple(sorted([zm[:2] + tuple(xgmap[p][0] for p in zm[2:]) for zm in gsm[ia]]))) if gsm is not None else gsm
                                xmr_eqs.append((gf, gt_unary_apply_xmap(xgmap), (gngs[ia], None), (gnm[ia], gnm[2]), gsm))
                        mf = sorted(xmr_eqs, key=lambda x: [x[1], x[3], x[2], abs(x[0])])[0][0]
                        xmr_eqs = sorted([(f / mf, meq, ngs, nm, xsm) for f, meq, ngs, nm, xsm in xmr_eqs], key=lambda x: (len(x[-2]), *x[::-1]))
                        xdp = None
                        for irx in range(pnx, len(lf_eqss)):
                            is_dup, fac = gt_meq_is_duplicate(xmr_eqs, lf_eqss[irx])
                            if is_dup:
                                xdp = irx - nouts, fac
                                break
                        if xdp is None:
                            xmr = "__%d" % nx
                            nx += 1
                            lf_eqss.append(xmr_eqs)
                            queue.append(len(lf_eqss) - 1)
                        else:
                            xmr = "__%d" % xdp[0]
                            mf *= xdp[1]
                        xsm = tuple(sorted([zm[:2] + tuple(xmap[p][0] for p in zm[2:]) for zm in sm[ia]])) if sm is not None else sm
                        if ia == 1:
                            new_lfeqs.append((mf, xmeq, (ngs[0], None, ngs[2]), (nm[0], xmr, nm[2]), (sm[0], xsm, sm[2]) if sm is not None else sm))
                        else:
                            new_lfeqs.append((mf, xmeq, (None, ngs[1], ngs[2]), (xmr, nm[1], nm[2]), (xsm, sm[1], sm[2]) if sm is not None else sm))
            changed = changed or len(vl) + len(vr) < len(conn)
        if changed:
            queue.append(queue[iq])
        lf_eqss[queue[iq]] = new_lfeqs
        iq += 1
    return gt_remove_unref(*gt_lfeqs_depend_reorder(nouts, lf_eqss[:pnx - nouts] + lf_eqss[pnx:] + lf_eqss[pnx - nouts:pnx]))

def gt_lfeqs_depend_reorder(nouts, lf_eqss):
    idx_mp = {}
    new_lf_eqss = []
    queue = [ix for ix in range(len(lf_eqss) - nouts)[::-1]]
    while len(queue) != 0:
        ix = queue[-1]
        if ix in idx_mp:
            queue.pop()
            continue
        ok = True
        for xxnm in lf_eqss[ix]:
            for x in xxnm[-2]:
                if x.startswith('__') and int(x[2:]) not in idx_mp:
                    ok = False
                    queue.append(int(x[2:]))
        if ok:
            if ix < len(lf_eqss) - nouts:
                new_lf_eqss.append(lf_eqss[ix])
            idx_mp[ix] = len(new_lf_eqss) - 1
            queue.pop()
    for ix in range(len(lf_eqss) - nouts, len(lf_eqss)):
        new_lf_eqss.append(lf_eqss[ix])
    for ix in range(len(new_lf_eqss)):
        for iq, xxnm in enumerate(new_lf_eqss[ix]):
            nnm = tuple('__%d' % idx_mp[int(x[2:])] if x.startswith('__') else x for x in xxnm[-2])
            new_lf_eqss[ix][iq] = xxnm[:-2] + (nnm, ) + xxnm[-1:]
    return nouts, new_lf_eqss

def gt_optimize_eqs(prim_eqs, meq_f, mani_f, symm_f=None, manifest=True, split=True, lift=True, sum_lift=True,
                    use_hint=True, label_symm=False, split_lift=1, df_compat=False, shapes=None):
    from functools import reduce
    import itertools
    if not split:
        return prim_eqs
    mn_eqs = []
    nx = 2 + any('#' in x[-2] or '^' in x[-2] for xs in prim_eqs for x in xs)
    for eqs in prim_eqs:
        xeqs = []
        mord = None
        for f, deq, n, name, deq_base in eqs:
            ord, assoc = gt_complexity(deq, n, name, deq_base, nx, shapes, use_hint=use_hint)
            deq, name, assoc = gt_reorder(deq, n, name, assoc, deq_base, nx)
            ex_eqs = gt_multi_extract(deq, n, name, deq_base, nx, assoc)
            ex_eqs = [(*meq_f(beq, len(nm), deq_base, nx, nbin=2), nm, assoc) for beq, nm, assoc in ex_eqs]
            ex_eqs = [gt_meq_reorder_using_nm(meq, len(nm), nm, name, ngs, '#', 1) + (assoc, ) for meq, ngs, nm, assoc in ex_eqs]
            ex_eqs = [gt_meq_reorder_using_nm(meq, len(nm), nm, name, ngs, '^', 1) + (assoc, ) for meq, ngs, nm, assoc in ex_eqs]
            ex_eqs = [gt_meq_reorder_using_nm(meq, len(nm), nm, name, ngs, 'G', 0) + (assoc, ) for meq, ngs, nm, assoc in ex_eqs]
            ex_eqs = [gt_meq_reorder_using_nm(meq, len(nm), nm, name, ngs, 'H', 0) + (assoc, ) for meq, ngs, nm, assoc in ex_eqs]
            ex_eqs = [(meq, [None if isinstance(xnm, tuple) or (ig == len(nm) - 1 and ix != len(ex_eqs) - 1) else ng for ig, (ng, xnm)
                in enumerate(zip(ngs, nm))], nm, assoc) for ix, (meq, ngs, nm, assoc) in enumerate(ex_eqs)]
            xeqs.append((symm_f(ex_eqs), ) if label_symm else ())
            if not lift and manifest:
                mani_ex_eqs = [(assoc, ) + mani_f(1.0, meq, len(nm), nm, ngs, nx, symms=xeqs[-1][0][im] if label_symm else None)
                    for im, (meq, ngs, nm, assoc) in enumerate(ex_eqs)]
                xeqs[-1] = (f * reduce(lambda x, y: x * y, [1] + [x[1] for x in mani_ex_eqs]),
                    [x[:1] + x[2:5] for x in mani_ex_eqs]) + (([x[-1] for x in mani_ex_eqs], ) if label_symm else ())
            else:
                xeqs[-1] = (f, ex_eqs) + xeqs[-1]
            ord = None if len(ord) == 0 else ord[0]
            if ord is not None and (mord is None or sum(ord) > sum(mord) or (sum(ord) == sum(mord) and ord > mord)):
                mord = ord
        mn_eqs.append(xeqs)
    if lift:
        lf_eqsss = [gt_lift(mn_eqs[l:r]) for l, r in zip([0] + list(range(1, split_lift + 1)), list(range(1, split_lift + 1)) + [None])]
        xlf_eqs = []
        for nouts, lf_eqss in lf_eqsss:
            if sum_lift:
                lf_eqss = gt_meq_sum_lift(lf_eqss, nouts)
            if manifest:
                lf_eqss = [[mani_f(f, meq, len(nm), nm, ngs, nx, symms=list(sm) if sm is not None else None)
                    for f, meq, ngs, nm, sm in lf_eqs] for lf_eqs in lf_eqss]
            iml, imdels = [-1] * len(lf_eqss), [[] for _ in lf_eqss] + [[]]
            for ix, lf_eqs in enumerate(lf_eqss):
                for x in lf_eqs:
                    for nmx in x[3][:-1]:
                        if nmx[0] == '_':
                            assert int(nmx[nmx.count('_'):]) < ix
                            iml[int(nmx[nmx.count('_'):])] = ix
            for ix, x in itertools.groupby(sorted(enumerate(iml), key=lambda x: x[1]), key=lambda x: x[1]):
                imdels[ix] = [xx[0] for xx in x]
            xlf_eqs.append((nouts, lf_eqss, imdels[:-1]))
        return xlf_eqs
    else:
        return mn_eqs
