
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

def gt_label_symm(ex_eqs):
    from pyhast.wick.gwick import gt_idx_dsu
    imd_symm = {}
    for meq, _, nms, assoc in ex_eqs:
        nc = max([0] + [1 | x[-1] for x in meq]) + 1
        imd_symm[assoc] = ()
        xmp = [{m[i + 1]: m[(i ^ 2) + 1] for m in meq for i in [0, 2] if m[i] == im and m[i ^ 2] == len(nms) - 1} for im in range(len(nms) - 1)]
        for im, nm in enumerate(nms[:-1]):
            if nm[0] in ['L', 'R', 'T']:
                for ic in sorted(range(nc), key=lambda x: [x & 1, x]):
                    ixs = [xmp[im][m[i + 1]] for m in meq for i in [0, 2] if m[i] == im and ic == 1 ^ m[4] and m[i + 1] in xmp[im]]
                    imd_symm[assoc] += ((-len(ixs), ic, *sorted(ixs)), )
            elif nm == 'H':
                ixs = sorted([(m[i + 1], 1 ^ m[4]) for m in meq for i in [0, 2] if m[i] == im])
                assert len(ixs) % 2 == 0
                for ki, kj in [(0, len(ixs) // 2), (len(ixs) // 2, len(ixs))]:
                    for ic in range(nc):
                        jxs = [xmp[im][ix] for ix, xc in ixs if ix >= ki and ix < kj and xc == ic and ix in xmp[im]]
                        imd_symm[assoc] += ((-len(jxs), ic, *sorted(jxs)), )
            else:
                for ixs in imd_symm[nm]:
                    jxs = sorted(xmp[im][p] for p in ixs[2:] if p in xmp[im])
                    imd_symm[assoc] += ((-len(jxs), ixs[1], *jxs), )
    for meq, ngs, nms, assoc in ex_eqs[-1:]:
        nc, icd = max([0] + [1 | x[-1] for x in meq]) + 1, len(ngs[-1]) - 2
        for ia, ib in [[0, icd], [icd, icd + ngs[-1][-2]], [icd + ngs[-1][-2], icd + ngs[-1][-2] + ngs[-1][-1]]]:
            for ic in sorted(range(nc), key=lambda x: [x & 1, x]):
                ixs = [m[i + 1] for m in meq for i in [0, 2] if m[i] == len(nms) - 1 and ic == 1 ^ m[4]]
                ixs = [x for x in ixs if x >= ia and x < ib]
                imd_symm[assoc] += ((-len(ixs), ic, *sorted(ixs)), )
    for meq, _, nms, assoc in ex_eqs[::-1]:
        xmp = [{m[i + 1]: m[(i ^ 2) + 1] for m in meq for i in [0, 2] if m[i] == len(nms) - 1 and m[i ^ 2] == im} for im in range(len(nms) - 1)]
        for im, nm in enumerate(nms[:-1]):
            if nm in imd_symm:
                for ixs in imd_symm[assoc]:
                    jxs = sorted(xmp[im][p] for p in ixs[2:] if p in xmp[im])
                    imd_symm[nm] += ((-len(jxs), ixs[1], *jxs), )
    for nm in imd_symm:
        cmap = {x: xsm[1] for xsm in imd_symm[nm] for x in xsm[2:]}
        conn = [x for xsm in imd_symm[nm] for x in ([(xsm[2], xsm[2])] if len(xsm[2:]) == 1 else zip(xsm[2:], xsm[3:]))]
        rsymms = [(-len(g), cmap[g[0]], *sorted(g)) for g in gt_idx_dsu(conn)]
        assert len(set(x for xsm in imd_symm[nm] for x in xsm[2:])) == len(set(x for xsm in rsymms for x in xsm[2:]))
        imd_symm[nm] = tuple(sorted(rsymms, key=lambda x: [x[1] & 1, x[1], x]))
    symms = []
    for meq, _, nms, assoc in ex_eqs:
        nc = max([0] + [1 | x[-1] for x in meq]) + 1
        xsymms = []
        for im, nm in enumerate(nms):
            xsymms.append(())
            if nm[0] in ['L', 'R', 'T']:
                for ic in sorted(range(nc), key=lambda x: [x & 1, x]):
                    ixs = [m[i + 1] for m in meq for i in [0, 2] if m[i] == im and ic == 1 ^ m[4]]
                    xsymms[im] += ((-len(ixs), ic, *sorted(ixs)), ) if len(ixs) != 0 else ()
            elif nm == 'H':
                ixs = sorted([(m[i + 1], 1 ^ m[4]) for m in meq for i in [0, 2] if m[i] == im])
                assert len(ixs) % 2 == 0
                for ki, kj in [(0, len(ixs) // 2), (len(ixs) // 2, len(ixs))]:
                    for ic in range(nc):
                        jxs = [ix for ix, xc in ixs if ix >= ki and ix < kj and xc == ic]
                        xsymms[im] += ((-len(jxs), ic, *sorted(jxs)), ) if len(jxs) != 0 else ()
            elif nm == 'X':
                xsymms[im] = imd_symm[assoc]
            else:
                xsymms[im] = imd_symm[nm]
        symms.append(xsymms)
    return symms

def gt_symm_take(x, symm_t, shapes, masks, addrs):
    import numpy as np
    addr, shape = (), ()
    x = x.transpose(tuple(p for x in symm_t for p in x[2:]))
    for it, (ov, na) in enumerate([(x[1], -x[0]) for x in symm_t if x[0] != 0]):
        x = x[(slice(None), ) * it + tuple(masks[na][ov])]
        addr = tuple(ix[..., None] for ix in addr) + (addrs[na][ov][(None, ) * it], )
        shape = shape + (shapes[na][ov], )
    xx = np.zeros(shape, dtype=x.dtype)
    xx[addr] = x
    return xx[tuple([None, slice(None)][x[0] != 0] for x in symm_t)]

def gt_symm_untake(x, symm_t, shapes, masks, addrs):
    import numpy as np
    rx = x[tuple(slice(None) if x[0] != 0 else 0 for x in symm_t)]
    nt, xx = 0, rx
    for it, (ov, na) in enumerate([(x[1], -x[0]) for x in symm_t if x[0] != 0]):
        z = np.zeros(xx.shape[:nt] + (shapes[1][ov], ) * na + rx.shape[it + 1:])
        z[(slice(None, ), ) * nt + tuple(masks[na][ov])] = xx[(slice(None, ), ) * nt + (addrs[na][ov], )]
        xx, nt = z, nt + na
    xx = xx.transpose(*np.argsort(tuple(p for x in symm_t for p in x[2:])))
    return gt_antisymm_sum(xx, symm_t)

def gt_antisymm_sum(x, symm_t):
    import itertools, numpy as np, pyhast.wick.gwick
    pord = tuple(p for x in symm_t for p in x[2:])
    rord = tuple(ip for _, ip in sorted([(p, ip) for ip, p in enumerate(pord)]))
    x = x.transpose(pord)
    ix = 0
    for ov, na in [(x[1], -x[0]) for x in symm_t]:
        if ov != -1:
            gx = np.zeros_like(x)
            psg = [(p, pyhast.wick.gwick.get_perm_sign(p)) for p in itertools.permutations(range(na))]
            lt, rt = range(0, ix), range(ix + na, x.ndim)
            for lit, lsg in psg:
                gx += lsg * x.transpose(*lt, *[il + ix for il in lit], *rt)
            x = gx
        ix += na
    return x.transpose(rord)

def gt_symm_contract(merge_schemes, script, symms, *tensors):
    import numpy as np, math, pyhast.wick.gwick
    scrs = script.split('->')[0].split(',')
    idx_mps = [{rr: ii for ii, rr in enumerate(scr)} for scr in scrs]
    out_mp = {rr: ii for ii, rr in enumerate(script.split('->')[1])}
    ctr_idx_mps = [{}, {}]
    ptensors = list(tensors)
    out_symm, ff = [], 1
    for it, (ts, symm) in enumerate(zip(tensors, symms)):
        dp = 0
        for ip, xsymm in enumerate(symm):
            scr_other = idx_mps[1 - it] if len(scrs) == 2 else {}
            ctr_idx = [ix for ix, p in enumerate(xsymm[2:]) if scrs[it][p] in scr_other]
            nctr_idx = [ix for ix, p in enumerate(xsymm[2:]) if scrs[it][p] not in scr_other]
            ctr_k = [scrs[it][xsymm[2 + ix]] for ix in ctr_idx]
            ctr_ord = [x[1] for x in sorted([(k, ik) for ik, k in enumerate(ctr_k)])]
            ff *= pyhast.wick.gwick.get_perm_sign(ctr_idx + nctr_idx) * pyhast.wick.gwick.get_perm_sign(ctr_ord)
            nctr = len(ctr_idx)
            if len(ctr_k) != 0:
                ctr_idx_mps[it][''.join(sorted(ctr_k))] = ip + dp
            if nctr > 0:
                if nctr < -xsymm[0]:
                    out_symm.append((-(-xsymm[0] - nctr), xsymm[1], *[out_mp[scrs[it][xsymm[2 + i]]] for i in nctr_idx]))
                    xsign, xaddr = merge_schemes[xsymm[1]][(-xsymm[0], nctr)]
                    if ts.size != 0:
                        ts = ts[(slice(None), ) * (ip + dp) + (xaddr, ) + (slice(None), ) * (ts.ndim - ip - dp - 1)] * \
                            xsign[(None, ) * (ip + dp) + (slice(None), ) * 2 + (None, ) * (ts.ndim - ip - dp - 1)]
                    else:
                        ts = np.zeros(ts.shape[:ip + dp] + xaddr.shape + ts.shape[ip + dp + 1:], dtype=ts.dtype)
                otix = [ixx for ix in ctr_k for ixx, xm in enumerate(symms[1 - it]) if scr_other[ix] in xm[2:]]
                if len(set(otix)) > 1:
                    del ctr_idx_mps[it][''.join(sorted(ctr_k))]
                    ctr_us = [ix for iu in sorted(set(otix)) for ix, iti in zip(ctr_k, otix) if iti == iu]
                    ctru_ord = [x[1] for x in sorted([(k, ik) for ik, k in enumerate(ctr_us)])]
                    ff *= pyhast.wick.gwick.get_perm_sign(ctru_ord)
                    xnctr = nctr
                    for iu in sorted(set(otix)):
                        ctr_u = [ix for ix, iti in zip(ctr_k, otix) if iti == iu]
                        ctr_idx_mps[it][''.join(sorted(ctr_u))] = ip + dp
                        if xnctr != len(ctr_u):
                            xsign, xaddr = merge_schemes[xsymm[1]][(xnctr, len(ctr_u))]
                            if ts.size != 0:
                                ts = ts[(slice(None), ) * (ip + dp) + (xaddr, ) + (slice(None), ) * (ts.ndim - ip - dp - 1)] * \
                                    xsign[(None, ) * (ip + dp) + (slice(None), ) * 2 + (None, ) * (ts.ndim - ip - dp - 1)]
                            else:
                                ts = np.zeros(ts.shape[:ip + dp] + xaddr.shape + ts.shape[ip + dp + 1:], dtype=ts.dtype)
                            dp += 1
                        xnctr -= len(ctr_u)
                if nctr < -xsymm[0]:
                    dp += 1
            elif nctr == 0:
                out_symm.append((xsymm[0], xsymm[1], *[out_mp[scrs[it][p]] for p in xsymm[2:]]))
        ptensors[it] = ts
    assert sorted(ctr_idx_mps[0].keys()) == sorted(ctr_idx_mps[1].keys())
    for k in ctr_idx_mps[0]:
        ff *= math.factorial(len(k))
    for xsm in out_symm:
        ff *= math.factorial(-xsm[0])
    for xsm in symms[-1]:
        ff /= math.factorial(-xsm[0])
    ctr_idxs = [tuple(ia for _, ia in sorted(mp.items())) for mp in ctr_idx_mps]
    rtensor = np.tensordot(ptensors[0], ptensors[1], axes=ctr_idxs) if len(scrs) == 2 else ptensors[0]
    out_idx_mp_sgrp = {p: ix for ix, xsm in enumerate(out_symm) for p in xsm[2:]}
    out_trs, out_fidxs, out_shapes = [], [], []
    merge_trs, nmerge_trs = [], []
    n_merge = 0
    for ip, xsymm in enumerate(symms[-1]):
        iqs = sorted(set(out_idx_mp_sgrp[p] for p in xsymm[2:]))
        out_trs.extend(iqs)
        assert len(iqs) in [1, 2]
        for psm in [xsymm[2:], [iqv for iqx in iqs for iqv in out_symm[iqx][2:]]]:
            ff *= pyhast.wick.gwick.get_perm_sign([x[1] for x in sorted([(k, ik) for ik, k in enumerate(psm)])])
        if len(iqs) == 1:
            nmerge_trs.append(iqs[0])
            out_fidxs.append(slice(None))
            out_shapes.append(rtensor.shape[iqs[0]])
        else:
            assert out_symm[iqs[0]][0] + out_symm[iqs[1]][0] == xsymm[0]
            xsign, xaddr = merge_schemes[xsymm[1]][(-xsymm[0], -out_symm[iqs[0]][0])]
            rtensor = xsign[(None, ) * iqs[0] + (slice(None), ) + (None, ) * (iqs[1] - iqs[0] - 1) +
                (slice(None), ) + (None, ) * (rtensor.ndim - iqs[1] - 1)] * rtensor
            out_fidxs.append(xaddr)
            out_shapes.append(np.max(xaddr) + 1 if xaddr.size != 0 and np.sum(np.abs(xsign)) != 0 else 0)
            n_merge += 1
            merge_trs.extend(iqs)
    if n_merge == 0:
        out_tensor = ff * rtensor.transpose(tuple(out_trs))
    else:
        out_tensor = np.zeros(tuple(out_shapes), dtype=rtensor.dtype)
        xofs = [iof for iof, oidx in enumerate(out_fidxs) if not isinstance(oidx, slice)]
        assert len(xofs) == n_merge
        out_trs = out_trs if max(xofs) - min(xofs) == len(xofs) - 1 else merge_trs + nmerge_trs
        for im, iof in enumerate(xofs):
            out_fidxs[iof] = out_fidxs[iof][(None, ) * (im * 2) + (slice(None), ) * 2 + (None, ) * ((n_merge - im - 1) * 2)]
        if out_tensor.size != 0:
            np.add.at(out_tensor, tuple(out_fidxs), ff * rtensor.transpose(tuple(out_trs)))
    return out_tensor
