
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

from pyhast.wick.gsymm import gt_symm_take, gt_symm_untake

import numpy as np
import math

def _pop_count_64(x):
    x = (x & 0x5555555555555555) + ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x & 0x0F0F0F0F0F0F0F0F) + ((x >> 4) & 0x0F0F0F0F0F0F0F0F)
    return (x * 0x0101010101010101) >> 56

def pop_count(arr, mask=None):
    x = arr.view(np.uint64)
    if mask is not None:
        x = x & np.asarray(mask, dtype=np.uint64)
    return (_pop_count_64(x[..., 0]) + _pop_count_64(x[..., 1])).astype(np.int64)

def get_ci_strings(n_occ, na):
    n_strs = math.comb(n_occ, na)
    r = np.zeros((n_strs, 2), dtype=np.int64)
    rem = np.arange(n_strs, dtype=np.int64)
    for k in range(na, 0, -1):
        v_candidates = np.arange(k - 1, n_occ, dtype=np.int64)
        binoms = np.array([math.comb(v, k) for v in v_candidates], dtype=np.int64)
        idx = np.searchsorted(binoms, rem, side='right') - 1
        vals = v_candidates[idx]
        rem -= binoms[idx]
        r[vals < 64, 0] |= (1 << vals[vals < 64])
        r[vals >= 64, 1] |= (1 << (vals[vals >= 64] - 64))
    return r

def strs_to_addr(n_occ, na, strs):
    weights = np.array([[math.comb(i, j) for j in range(na + 1)] for i in range(n_occ)], dtype=np.int64)
    strs_u = strs.view(np.uint64)
    bits = []
    for k in range((n_occ + 63) // 64):
        bits.append((strs_u[:, k:k+1] >> np.arange(min(64, n_occ - k * 64), dtype=np.uint64)) & 1)
    bits = np.concatenate(bits, axis=1).astype(np.int64) if bits else np.zeros((strs.shape[0], 0), dtype=np.int64)
    return np.sum(weights[np.arange(n_occ), np.cumsum(bits, axis=1)] * bits, axis=1)

def ut_symm_schemes(n_occ, n_vir, t_ord, n_cas=(0, 0)):
    n_fci_ts, cistrs = [], [[], [], [], [], [], []]
    nns = [n_occ[0], n_vir[0], n_occ[1], n_vir[1], n_cas[0], n_cas[1]]
    for na in range(t_ord + 1):
        n_fci_ts.append(())
        for ix, nn in enumerate(nns):
            n_fci_ts[na] += (math.comb(nn, na), )
            cistrs[ix].append(get_ci_strings(nn, na))
    tamp_addrs, tamp_masks = [[()[:0] for _ in range(t_ord + 1)] for _ in range(2)]
    for na, xshapes in enumerate(n_fci_ts):
        for nn, xcistr, xshape in zip(nns, cistrs, xshapes):
            cur_strs = xcistr[na].view(np.uint64)
            bits = []
            for k in range((nn + 63) // 64):
                bits.append((cur_strs[:, k:k+1] >> np.arange(min(64, nn - k * 64), dtype=np.uint64)) & 1)
            bits = np.concatenate(bits, axis=1) if bits else np.zeros((xcistr[na].shape[0], 0), dtype=np.uint64)
            xmask = np.nonzero(bits)[1].reshape(xcistr[na].shape[0], na).T if bits.size and na > 0 else np.zeros((na, xcistr[na].shape[0]), dtype=np.int64)
            tamp_addrs[na] += (np.mgrid[:xshape], )
            tamp_masks[na] += (xmask, )
    return (n_fci_ts, tamp_masks, tamp_addrs), cistrs

def ut_merge_schemes(n_occ, n_vir, t_ord, cistrs):
    rs = [{}, {}, {}, {}]
    def make_mask(nn):
        m = np.zeros(2, dtype=np.uint64)
        if nn <= 64:
            m[0] = (1 << nn) - 1
        else:
            m[0], m[1] = 0xFFFFFFFFFFFFFFFF, (1 << (nn - 64)) - 1
        return m
    for ov, (nn, cistr) in enumerate(zip([n_occ[0], n_vir[0], n_occ[1], n_vir[1]], cistrs)):
        full_mask = make_mask(nn)
        for na in range(t_ord + 1):
            for nb in range(na + 1):
                tl, tr = cistr[nb][:, None, :], cistr[na - nb][None, :, :]
                sq = tl | tr
                mask = pop_count(sq, full_mask) == na
                sq_flat = sq[mask]
                xaddrs = np.zeros(sq.shape[:2], dtype=np.int64)
                if sq_flat.size:
                    xaddrs[mask] = strs_to_addr(nn, na, sq_flat)
                phase_acc = np.zeros(mask.shape, dtype=np.int64)
                tl_u, tr_u = tl.view(np.uint64), tr.view(np.uint64)
                for l in range(nn):
                    blk, rem = divmod(l, 64)
                    bit_l = (tl_u[..., blk] >> np.uint64(rem)) & 1
                    mask_val = (np.uint64(1) << np.uint64(rem)) - np.uint64(1)
                    tr_cnt = _pop_count_64(tr_u[..., blk] & mask_val)
                    if blk:
                        tr_cnt += _pop_count_64(tr_u[..., 0])
                    phase_acc += bit_l.astype(np.int64) * tr_cnt.astype(np.int64)
                xsg = 1 - ((phase_acc & 1) << 1)
                xsg[~mask] = 0
                rs[ov][(na, nb)] = xsg, xaddrs
    return rs

def ut_symm_amps(ts, symm_schemes, ipea=0):
    ts = [list(x) for x in ts]
    for xts in ts:
        for p, xt in enumerate(xts):
            nn, k, kk = (xt.ndim + abs(ipea)) // 2, p // (abs(ipea) + 1), p % (abs(ipea) + 1)
            nns = [0, nn - k - max(ipea, 0) - kk * (ipea < 0), nn - max(ipea, 0),
                nn * 2 - k - abs(ipea) - kk * (ipea > 0), nn * 2 - abs(ipea)]
            tsymm = tuple((a - b, p, *range(a, b)) for p, a, b in zip([0, 2, 1, 3], nns, nns[1:]) if a != b)
            xts[p] = gt_symm_take(xt, tsymm, *symm_schemes)
    return [list(x) for x in ts]

def ut_unsymm_amps(ts, symm_schemes, has_ref=False, ipea=0):
    ts = [list(x) for x in ts]
    for it, xts in enumerate(ts):
        nn = (it if has_ref else it + 1) + max(abs(ipea) - 1, 0)
        for p, xt in enumerate(xts):
            k, kk = p // (abs(ipea) + 1), p % (abs(ipea) + 1)
            nns = [0, nn - k - max(ipea, 0) - kk * (ipea < 0), nn - max(ipea, 0),
                nn * 2 - k - abs(ipea) - kk * (ipea > 0), nn * 2 - abs(ipea)]
            tsymm = tuple((a - b, p, *range(a, b)) for p, a, b in zip([0, 2, 1, 3], nns, nns[1:]) if a != b)
            xts[p] = gt_symm_untake(xt, tsymm, *symm_schemes)
    return [list(x) for x in ts]

def ut_unsymm_npdm(r, tag, symm_schemes):
    tags = [(0, tag[:len(tag) // 2]), (len(tag) // 2, tag[len(tag) // 2:])]
    idxs = [[it + ix for it, t in enumerate(xtag) if t == a] for ix, xtag in tags for a in 'iIeE']
    rsymm = tuple((-len(x), ic, *x) for ic, x in zip([0, 2, 1, 3] * 2, idxs) if len(x) != 0)
    rsymm = tuple(sorted(rsymm, key=lambda x: [x[1] & 1, x[1], x]))
    return gt_symm_untake(r, rsymm, *symm_schemes)

def ut_symm_integral(ints, n_occ, symm_amps, symm_schemes):
    import numpy as np, itertools
    sm_ints = [{} for _ in range(len(ints))]
    nocca, noccb = n_occ
    sli = [[slice(nocca), slice(nocca, None)], [slice(noccb), slice(noccb, None)]]
    for ndd, f in itertools.groupby(zip(ints, sm_ints), key=lambda x: x[0].ndim):
        icd, nd = ndd % 2, ndd // 2 * 2
        for it, (xint, smint) in enumerate([(a, b) for a, b in f if not icd or a.shape[0] != 0]):
            for ix in np.mgrid[(slice(2), ) * nd].reshape((nd, -1)).T:
                zidx = 'G' * icd + ''.join(['ie', 'IE'][ip % (nd // 2) >= nd // 2 - it][x] for ip, x in enumerate(ix))
                zint = xint[(slice(None), ) * icd + tuple(sli[ip % (nd // 2) >= nd // 2 - it][x] for ip, x in enumerate(ix))]
                if symm_amps:
                    assert nd % 2 == 0
                    zsymm = ()
                    for ki, kj in [(0, nd // 2), (nd // 2, nd)]:
                        for ic in range(4):
                            ps = [iz + ki for iz, xc in enumerate(zidx[ki:kj]) if xc == 'ieIE'[ic]]
                            zsymm += ((-len(ps), ic, *sorted(ps)), ) if len(ps) != 0 else ()
                    zint = gt_symm_take(zint, zsymm, *symm_schemes)
                smint[zidx] = zint
    return sm_ints
