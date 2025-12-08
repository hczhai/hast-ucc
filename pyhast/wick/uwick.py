
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

def ut_differentiate(eq, nts, names, is_mr=False, is_hbar=False, deq_base=4, nx=2):
    from pyhast.wick.gwick import gt_notes
    mt = gt_notes(is_mr, is_hbar)
    ts = [[x, x] for n in range(nts) for x in [sum((x >> n & 1) * (1 + (x == (1 << n))) for x in eq) // 2]]
    def deq_search(ictr, pctr, r=[], deq=[0] * len(eq), ss=[[0, 0] for _ in range(nts)]):
        if ictr == len(eq):
            if all(a == b for a, b in ss):
                r.append(tuple(sorted(deq)))
        else:
            ij = [x for x in range(nts) if eq[ictr] & (1 << x)]
            i, j = ij if len(ij) == 2 else ij + ij
            abcgs = [(a, b, c, g) for c in [0, 1] for a, b, g in mt.get(names[i] + names[j], mt[''])]
            for ip, (a, b, c, g) in enumerate(abcgs[pctr:]):
                if ts[i][a] > 0 and ts[j][b] > 0 and (i < j or a > b):
                    ts[i][a], ts[j][b] = ts[i][a] - 1, ts[j][b] - 1
                    ss[i][a], ss[j][b] = ss[i][a] + c, ss[j][b] + c
                    deq[ictr] = (1 << (i * deq_base * nx + c * nx * 2 + g + b * nx)) | (1 << (j * deq_base * nx + c * nx * 2 + g + a * nx))
                    npctr = 0 if ictr == len(eq) - 1 or eq[ictr + 1] != eq[ictr] else pctr + ip
                    deq_search(ictr + 1, npctr, r, deq)
                    ts[i][a], ts[j][b] = ts[i][a] + 1, ts[j][b] + 1
                    ss[i][a], ss[j][b] = ss[i][a] - c, ss[j][b] - c
        return r
    return sorted(deq_search(0, 0))

def ut_quantify(eq, nts, names, deq_base, nx):
    ts, ctr_idxs, acc_sign = [0] * nts, [], 0
    ng = max([sum(1 for x in eq for i in range(deq_base * nx) if x & (1 << (n * deq_base * nx + i))) for n in range(nts)])
    for x in eq: # E+(0) I+(1) I(2) E(3)
        zcd, xcd = [(n, i % nx, i // nx % 2, i // (nx * 2)) for n in range(nts) for i in range(deq_base * nx) if x & (1 << (n * deq_base * nx + i))], [0, 0]
        for icd, zx in enumerate(zcd if len(zcd) == 2 else zcd + zcd):
            xcd[icd] = ts[zx[0]] + zx[0] * ng * 4 * nx + zx[3] * ng + ng * 2 * nx * zx[2] + ng * 2 * (zx[1] == zx[2]
                if '#' not in names and '^' not in names else (zx[1] + 1 - zx[2]) % 3)
            ts[zx[0]] += 1 # E(i=0) I+(i=1) [icd=0] E+(i=0) I(i=1) [icd=1]
        for xc, xd, xa, xb in [(xc, xd, xa, xb) for xc, xd in [xcd] for xa, xb in ctr_idxs]:
            acc_sign ^= ((xa < xc and xb > xc and xb < xd) or (xa > xc and xa < xd and xb > xd))
        ctr_idxs.append(tuple(xcd))
    acc_sign ^= sum(1 for x in ts if x % 8 >= 4) & 1 # v[pq..rs..] pq.. ..sr
    factor, acc = 1.0, 1
    for a, b in zip(eq, eq[1:]):
        acc = acc * (a == b) + 1
        factor /= acc
    return -factor if acc_sign else factor

def ut_adapt_ipea(eq, nts, names, deq_base, nx, ir, ix, ipea=0):
    import math
    from pyhast.wick.gwick import get_perm_sign
    nipea = [[0, 0], [0, 0]]
    for x in eq:
        zcd = [(n, i % nx, i // nx % 2, i // (nx * 2)) for n in range(nts) for i in range(deq_base * nx) if x & (1 << (n * deq_base * nx + i))]
        ok = len(zcd) == 2 and ((zcd[0][0] == ir and zcd[1][0] == ix) or (zcd[0][0] == ix and zcd[1][0] == ir))
        ic, ip = (zcd[zcd[1][0] == ix][2] if ok else 0), zcd[0][3]
        nipea[ic][ip] += ok
    reqs, (na, nb) = [], (nipea[ipea < 0][0], nipea[ipea < 0][1])
    if ipea == 0 or na + nb < abs(ipea):
        return [(1, eq)] if ipea == 0 else []
    ng = max([sum(1 for x in eq for i in range(deq_base * nx) if x & (1 << (n * deq_base * nx + i))) for n in range(nts)])
    for ia, ib in [(ia, ib) for ia in range(na + 1) for ib in [abs(ipea) - ia] if ib >= 0 and ib <= nb]:
        neq, ja, jb = [], 0, 0
        ts, trs, tks = [0] * nts, [[] for _ in range(nts)], [[] for _ in range(nts)]
        for x in eq:
            zcd, xcd = [(n, i % nx, i // nx % 2, i // (nx * 2)) for n in range(nts) for i in range(deq_base * nx) if x & (1 << (n * deq_base * nx + i))], []
            for zx in (zcd if len(zcd) == 2 else zcd + zcd):
                xcd.append((zx[0], ts[zx[0]] + zx[0] * ng * 4 * nx + zx[3] * ng + ng * 2 * nx * zx[2] + ng * 2 * (zx[1] == zx[2]
                    if '#' not in names and '^' not in names else (zx[1] + 1 - zx[2]) % 3)))
                ts[zx[0]] += 1
            ok = len(zcd) == 2 and ((zcd[0][0] == ir and zcd[1][0] == ix) or (zcd[0][0] == ix and zcd[1][0] == ir))
            ic, ip = (zcd[zcd[1][0] == ix][2] if ok else 0), zcd[0][3]
            if not (ok and ((ja < ia and zcd[0][3] == 0) or (jb < ib and zcd[0][3] == 1)) and ic == int(ipea < 0)):
                neq.append(x)
                for ixx, x in xcd:
                    tks[ixx].append(x)
            else:
                for ixx, x in xcd:
                    trs[ixx].append(x)
                ja, jb = ja + (ip == 0), jb + (ip == 1)
        df = get_perm_sign(sorted(trs[ir]) + sorted(tks[ir])) * get_perm_sign(sorted(trs[ix]) + sorted(tks[ix]))
        assert ja == ia and jb == ib
        reqs.append((df * math.factorial(na) * math.factorial(nb) / (math.factorial(na - ia) * math.factorial(nb - ib)), tuple(neq)))
    return reqs

def ut_spin_adjustment(factor, script, idxs, names, ngs, symms=None):
    scripts = script.split('->')[0].split(',') + script.split('->')[1:]
    cxl = {'e': 0, 'a': 1, 'i': 2, 'E': 3, 'A': 4, 'I': 5, 'G': 6}
    cxr = {'i': 0, 'a': 1, 'e': 2, 'I': 3, 'A': 4, 'E': 5, 'G': 6}
    for ix, (scr, idx, ng) in enumerate(zip(scripts, idxs, ngs)):
        xscr, xidx, icd = list(scr), list(idx), (0 if ng is None else len(ng) - 2)
        ils = [] if ng is None else [(icd, ng[-2], cxl), (icd + ng[-2], ng[-1], cxr)]
        for il, j, cx in [(il, j, cx) for il, l, cx in ils for i in range(l - 1) for j in range(l - 1, i, -1)]:
            if cx[xidx[il + j - 1]] > cx[xidx[il + j]]:
                xidx[il + j - 1], xidx[il + j] = xidx[il + j], xidx[il + j - 1]
                xscr[il + j - 1], xscr[il + j] = xscr[il + j], xscr[il + j - 1]
                factor *= -1
        scripts[ix], idxs[ix] = ''.join(xscr), ''.join(xidx)
        if symms is not None:
            symms[ix] = tuple(xsm[:2] + tuple(xscr.index(scr[x]) for x in xsm[2:]) for xsm in symms[ix])
    return (factor, ','.join(scripts[:-1]) + '->' + scripts[-1], idxs, names) + ((symms, ) if symms is not None else ())

def ut_real_hermitian(eq, nts, names, deq_base, nx, real_names):
    if real_names is None:
        return 1, eq
    ts = [([], []) for _ in range(nts)]
    for n, idx, dag in [(n, i % nx + i // (nx * 2) * nx, i // nx % 2) for x in eq for n in range(nts)
        for i in range(deq_base * nx) if x & (1 << (n * deq_base * nx + i))]:
        ts[n][dag].append(idx)
    fs = [int(nm[0] in real_names and sorted(ts[ix][1]) > sorted(ts[ix][0])) for ix, nm in enumerate(names)]
    f = -1 if sum([len(a) & len(b) for ix in range(nts) if fs[ix] for a, b in [ts[ix]]]) % 2 else 1
    return f, tuple(sum([1 << (n * deq_base * nx + i % nx + i // (nx * 2) * (nx * 2) + (fs[n] ^ (i // nx % 2)) * nx) for n in range(nts)
        for i in range(deq_base * nx) if x & (1 << (n * deq_base * nx + i))]) for x in eq)

def ut_deq_to_meq(eq, nts, deq_base, nx, nbin=2):
    ts = [[[] for _ in range(deq_base * nx)] for _ in range(nts * nbin)]
    for ix, x in enumerate(eq):
        zc, zd = [(n, i % nx, i // nx % 2, i // (nx * 2)) for n in range(nts * nbin) for i in range(deq_base * nx) if x & (1 << (n * deq_base * nx + i))]
        ts[zc[0]][zc[3] + zc[2] * 2 * nx + (zc[1] == zc[2] if nx == 2 else (zc[1] + 1 - zc[2]) % 3) * 2].append((ix, 0))
        ts[zd[0]][zd[3] + zd[2] * 2 * nx + (zd[1] == zd[2] if nx == 2 else (zd[1] + 1 - zd[2]) % 3) * 2].append((ix, 1))
    ngs = [[sum([len(x) for xts in ts[i:i + nbin] for x in xts[it * 2 * nx:(it + 1) * 2 * nx]]) for it in range(2)]
        for i in range(0, nts * nbin, nbin)]
    ts = [[xx for gx in ts[i:i + nbin] for xx in gx] for i in range(0, nts * nbin, nbin)]
    vs = [[-1, -1] for _ in eq]
    for p in range(nts):
        ip = 0
        for xx in ts[p]:
            for (ix, dx) in xx:
                vs[ix][dx] = ip
                ip += 1
    meq = []
    for ix, (x, v) in enumerate(zip(eq, vs)):
        zc, zd = [(n, i % nx, i // (nx * 2)) for n in range(nts * nbin) for i in range(deq_base * nx) if x & (1 << (n * deq_base * nx + i))]
        assert zc[1] == zd[1] and zc[2] == zd[2]
        meq.append((zc[0] // nbin, v[0], zd[0] // nbin, v[1], zc[1] + zc[2] * nx))
    return tuple(sorted(meq)), ngs

def ut_meq_manifest(f, meq, nts, names, ngs, nx, symms=None):
    from pyhast.wick.gwick import get_letter_index
    nd = max([0] + [max(ia, ib) for _, ia, _, ib, _ in meq]) + 2
    ctr_idx, ts = [0] * nx, [[""] * nd for _ in range(nts)]
    xs = [[""] * nd for _ in range(nts)]
    ref = list('abcdefghABCDEFGHrstuvwxyRSTUVWXYijklmnopqIJKLMNOPQZz0123456789!\"#$%&\'()*+./:;<=?@[\\]^_`{|}~')
    xptr = [ref.index('a')] + ([] if nx == 2 else [ref.index('r')]) + [ref.index('i'), ref.index('Z')]
    idxd = [{}, {}, {}, {}]
    fx = ['eE', 'iI'] if nx == 2 else ['eE', 'aA', 'iI']
    for a, ia, b, ib, c in meq:
        ts[a][ia] += get_letter_index(ref, xptr, idxd, c % nx, ctr_idx[c % nx])
        ts[b][ib] += get_letter_index(ref, xptr, idxd, c % nx, ctr_idx[c % nx])
        xs[a][ia] += fx[c % nx][c // nx]
        xs[b][ib] += fx[c % nx][c // nx]
        ctr_idx[c % nx] = ctr_idx[c % nx] + 1
    ngs = [None if ng is None else [1][:ig] + [ng[0] - ig] + ng[1:] for it, ng in enumerate(ngs) for ig in [0]]
    script = ','.join(''.join(p for xx in x for p in xx) for nm, x in zip(names, ts) if nm != 'X') + '->'
    script += ''.join(p for xx in ts[names.index('X')] for p in xx)
    idxs = [''.join(p for xx in x for p in xx) for x in [x for nm, x in zip(names, xs) if nm != 'X'] + [xs[names.index('X')]]]
    return ut_spin_adjustment(f, script, idxs, [nm for nm in names if nm != 'X'] + ['X'],
        [g for nm, g in zip(names, ngs) if nm != 'X'] + [ngs[names.index('X')]], symms=symms)

def ut_manifest(f, eq, nts, names, deq_base, nx, nbin=1):
    from pyhast.wick.gwick import get_letter_index
    ctr_idx, ts, is_mr = [0] * nx, [[""] * deq_base * nx for _ in range(nts * nbin)], '#' in names or '^' in names
    ref = list('abcdefghABCDEFGHrstuvwxyRSTUVWXYijklmnopqIJKLMNOPQZz0123456789!\"#$%&\'()*+./:;<=?@[\\]^_`{|}~')
    xptr = [ref.index('a')] + ([] if not is_mr else [ref.index('r')]) + [ref.index('i'), ref.index('Z')]
    idxd = [{}, {}, {}, {}]
    fx = lambda p: [x for x, d in zip(['eE', 'iI', 'gG'] if not is_mr else ['eE', 'aA', 'iI', 'gG'], idxd) if p in d.values()][0]
    ps = [n for x in eq for n in range(nts * nbin) for i in range(nx - 1, deq_base * nx, nx)
        if x & (1 << (n * deq_base * nx + i))] if nx - is_mr != 2 else []
    for x in eq:
        zcd = [(n, i % nx, i // nx % 2, i // (nx * 2)) for n in range(nts * nbin) for i in range(deq_base * nx) if x & (1 << (n * deq_base * nx + i))]
        for zc, zd in ([zcd] if len(zcd) == 2 else [zcd + zcd]):
            ic = zc[3] + zc[2] * 2 * nx + (zc[1] == zc[2] if not is_mr else (zc[1] + 1 - zc[2]) % 3) * 2
            id = zd[3] + zd[2] * 2 * nx + (zd[1] == zd[2] if not is_mr else (zd[1] + 1 - zd[2]) % 3) * 2
            ts[zc[0]][ic if zc[0] not in ps else (ic + 1 if zc[1] != nx - 1 else 0)] += get_letter_index(ref, xptr, idxd, zc[1], ctr_idx[zc[1]])
            ts[zd[0]][id if zd[0] not in ps else (id + 1 if zd[1] != nx - 1 else 0)] += get_letter_index(ref, xptr, idxd, zd[1], ctr_idx[zd[1]])
            ctr_idx[zc[1]], ctr_idx[zd[1]] = ctr_idx[zc[1]] + 1, ctr_idx[zd[1]] + 1
    if nbin != 1:
        ts = [[y for x in ts[i:i + nbin] for y in x] for i in range(0, nts * nbin, nbin)]
    ngs = [[1][:ig] + [sum([len(x) for x in xts[ig + it * 2 * nx:ig + (it + 1) * 2 * nx]]) for it in range(2)]
        for it, xts in enumerate(ts) for ig in [it in ps]]
    script = ','.join(''.join(p for xx in x for p in xx) for nm, x in zip(names, ts) if nm != 'X') + '->'
    script += ''.join(p for xx in ts[names.index('X')] for p in xx)
    idxs = [''.join(fx(p)[ix % 2] for ix, xx in enumerate(x) for p in xx)
        for x in [x for nm, x in zip(names, ts) if nm != 'X'] + [ts[names.index('X')]]]
    return ut_spin_adjustment(f, script, idxs, [nm for nm in names if nm != 'X'] + ['X'],
        [g for nm, g in zip(names, ngs) if nm != 'X'] + [ngs[names.index('X')]])

def ut_simplify_eqs(t_ord, prim_eqs, x_ord=None, manifest=True, filter_x=True, real_names=None, is_hbar=False,
                    ts_map=None, adapt_df=True, adapt_ipea=0, ipea_names='RX'):
    import itertools
    from pyhast.wick.gwick import gt_canonicalize

    x_ord = t_ord if x_ord is None else x_ord
    deq_base = 4
    nx = 2 + any('#' in x[-1] or '^' in x[-1] for x in prim_eqs)

    all_eqs, new_eqs = [], {(x[3], nctr): [] for x in prim_eqs for nctr in x[2]}
    for pref, eqs, nctrs, n, names in prim_eqs:
        for eq, nctr in zip(eqs, nctrs):
            xeqs, xnames = zip(*[gt_canonicalize(q, len(names), names) for q in eq])
            new_eqs[(n, nctr)].extend(zip(xeqs, xnames, [pref] * len(xnames)))
    for nq in new_eqs.values():
        uniq_eqs = [(*k, sum(g[-1] for g in f)) for k, f in itertools.groupby(sorted(nq), key=lambda x: x[:2])]
        nq = [(x, nm, f) for x, nm, f in uniq_eqs if abs(f) > 1E-14]
    for n, nqs in itertools.groupby(sorted(new_eqs.items()), key=lambda x: x[0][0]):
        all_eqs.append((n, *zip(*[zip(*nq) for _, nq in nqs if len(nq) != 0])))

    all_deqs = []
    for n, eqs, names, factors in all_eqs:
        deqs = []
        for fcs, eq, name in zip(factors, eqs, names):
            xdeqs = []
            for f, q, nm in zip(fcs, eq, name):
                pdeqs = [pdeq for pdeq in ut_differentiate(q, n, nm, deq_base=deq_base, is_mr='#' in nm or '^' in nm, is_hbar=is_hbar, nx=nx)]
                nm = [ts_map.get(xnm, xnm) for xnm in nm] if ts_map is not None else nm
                pdeqs = [ut_real_hermitian(deq, n, nm, deq_base, nx, real_names) for deq in pdeqs]
                xdeqs.extend([(*gt_canonicalize(q, n, nm, deq=deq, deq_base=deq_base, nx=nx), f * g) for g, deq in pdeqs])
            uniq_eqs = [(*k, sum(g[-1] for g in f)) for k, f in itertools.groupby(sorted(xdeqs), key=lambda x: x[:2])]
            deqs.extend([(f * ut_quantify(deq, n, nm, deq_base=deq_base, nx=nx), deq, nm) for deq, nm, f in uniq_eqs if abs(f) > 1E-14])
        deqs = [(f * g, neq, nm) for f, deq, nm in deqs for g, neq in ut_adapt_ipea(deq, n, nm, deq_base, nx,
            nm.index(ipea_names[0]), nm.index(ipea_names[1]), adapt_ipea)] if adapt_ipea != 0 else deqs
        all_deqs.append(deqs)

    tensor_eqs = [[] for _ in range(x_ord + 1)]
    for (n, eqs, names, factors), deqs in zip(all_eqs, all_deqs):
        for f, deq, name in deqs:
            idd = name.index('X')
            cnt = sum(1 for x in deq if any((x & (1 << (idd * deq_base * nx + i)) for i in range(deq_base * nx))))
            cnt = cnt + abs(adapt_ipea) if 'X' in ipea_names else cnt
            if manifest:
                tensor_eqs[cnt // 2].append(ut_manifest(f, deq, n, name, deq_base=deq_base, nx=nx))
            else:
                tensor_eqs[cnt // 2].append((f, deq, n, name, deq_base))
    return tensor_eqs

def ut_antisymmetrize(r, tag, is_ints=False, ipea=0):
    import itertools, numpy as np
    from pyhast.wick.gwick import get_perm_sign
    rr, gr, nl, nr = np.zeros_like(r), np.zeros_like(r), (r.ndim - ipea) // 2, (r.ndim + ipea) // 2
    psgl = [(p, get_perm_sign(p)) for p in itertools.permutations(range(nl))]
    psgr = psgl if nl == nr else [(p, get_perm_sign(p)) for p in itertools.permutations(range(nr))]
    tags = tag[:nl], tag[nl:]
    ptags = [["".join(xtag[x] for x in p) for p, _ in psg] for psg, xtag in zip([psgl, psgr], tags)]
    psgl, psgr = [[(p, s) for (p, s), t in zip(psg, xp) if t == xt] for psg, xt, xp in zip([psgl, psgr], tags, ptags)]
    for lit, lsg in psgl:
        gr += lsg * r.transpose(*lit, *range(nl, nl + nr))
    for rit, rsg in psgr:
        rr += rsg * gr.transpose(*range(nl), *[r + nl for r in rit])
    return rr / (len(psgl) * len(psgr)) if is_ints else rr

def ut_antisymmetrize_hast(r, tag):
    import itertools, numpy as np
    from pyhast.wick.gwick import get_perm_sign
    from pyhast.tensor.packed import PackedTensor
    rr, gr = PackedTensor(np.empty(r.shape)), PackedTensor(np.empty(r.shape))
    n = r.ndim // 2
    psg = [(p, get_perm_sign(p)) for p in itertools.permutations(range(n))]
    ptag = ["".join(tag[x] for x in p) for p, _ in psg]
    psg = [(p, s) for (p, s), t in zip(psg, ptag) if t == tag[:n]]
    assert len(psg) != 0
    for ir, (lit, lsg) in enumerate(psg):
        rx = PackedTensor(r.transpose(*lit, *range(n, n + n)))
        PackedTensor.add(rx, gr, alpha=lsg, beta=0.0 if ir == 0 else 1.0)
    for ir, (rit, rsg) in enumerate(psg):
        rx = gr.reorder((*range(n), *[r + n for r in rit]))
        PackedTensor.add(rx, rr, alpha=rsg, beta=0.0 if ir == 0 else 1.0)
    return rr.to_array()

def ut_npdm_denormal_order(dms, n_occ):
    import itertools, numpy as np
    from pyhast.wick.gwick import get_perm_sign
    dms = [[np.array(1.0)]] + dms
    for i in range(len(dms))[::-1]:
        for j in range(0, i):
            idx = tuple([np.mgrid[:n_occ[x]][(None, ) * k + (slice(None), ) + (None, ) * (i - j - k - 1)]
                for x in [0, 1]] for k in range(i - j)) + ([slice(None)] * 2, )
            perms = [[[x[1] for x in itertools.accumulate(range(i), lambda r, x: (r[0], -1) if x in pl else (r[0] + 1, px[r[0]]),
                initial=(0, ''))][1:] for pl in itertools.combinations(range(i), j)] for px in itertools.permutations(range(i - j))]
            for f, l, r in ((get_perm_sign(l) * get_perm_sign(r), l, r) for l in perms[0] for perm in perms for r in perm):
                hnone = [ix for ix, x in enumerate(l + r) if x != -1]
                if max(hnone) - min(hnone) + 1 != len(hnone):
                    hidx = tuple([None] * (i - j) + [slice(None) for x in l + r if x == -1])
                else:
                    hidx = tuple([None, slice(None)][x == -1] for ix, x in enumerate(l + r) if x == -1 or ix < i)
                for ii in range(i + 1):
                    if all((i - 1 - ix < ii) == (i - 1 - r.index(x) < ii) for ix, x in enumerate(l) if x != -1):
                        pj = [i - 1 - ix % i < ii for ix, x in enumerate(l + r) if x == -1]
                        if pj[:j] == pj[j:] and all(p <= q for p, q in zip(pj, pj[1:j])):
                            dms[i][ii][tuple(idx[x][i - 1 - ix % i < ii] for ix, x in enumerate(l + r))] \
                                += f * dms[j][pj[:j].count(True)][hidx]
    return dms[1:]

def ut_dfs_sum_lift_summary(lf_eqsss, n_occ, n_virt, n_cas):
    import hast
    from functools import reduce
    print("%5s %50s %10s %10s %14s" % ("order", "shape", "amp size", "work size", "cost"))
    for p, xeqs in enumerate(lf_eqsss):
        nouts, lf_eqss = xeqs[:2]
        for iout in range(-nouts, 0):
            shapes, sz, fl = ut_dfs_sum_lift_work_size(iout, lf_eqss, n_occ, n_virt, n_cas)
            szt = sum([reduce(lambda x, y: x * y, shape, 1) for shape in shapes])
            print("%5d %50s %10s %10s %14s" % (iout + nouts + p, shapes[0], hast.to_size_string(szt * 8),
                hast.to_size_string(sz * 8), hast.to_size_string(fl, 'FLOPs')))

def ut_dfs_sum_lift_work_size(iout, lf_eqss, n_occ, n_virt, n_cas):
    from functools import reduce
    sli = {'i': n_occ[0], 'a': n_cas[0], 'e': n_virt[0], 'I': n_occ[1], 'A': n_cas[1], 'E': n_virt[1]}
    prx = lambda idx: (idx.count('I') + idx.count('A') + idx.count('E')) // 2
    nwork, nflops = 0, 0
    nshapes = [()] * (max([0] + [prx(x[2][-1]) for x in lf_eqss[iout]]) + 1)
    for _, script, idxs, nm, *_ in lf_eqss[iout]:
        tensors = []
        ztensor = 0
        for nmx, idx in zip(nm[:-1], idxs[:-1]):
            if nmx[0] == '_':
                xshapes, xwork, xflops = ut_dfs_sum_lift_work_size(int(nmx[nmx.count('_'):]), lf_eqss, n_occ, n_virt, n_cas)
                tensors.append(xshapes[prx(idx)])
                ztensor += reduce(lambda x, y: x * y, xshapes[prx(idx)], 1)
                nwork = max(nwork, xwork + ztensor)
                nflops += xflops
            else:
                tensors.append(tuple(sli[x] for x in idx))
        shmap = {k: s for ten, scr in zip(tensors, script.split('->')[0].split(',')) for k, s in zip(scr, ten)}
        nflops += reduce(lambda x, y: x * y, shmap.values(), 1)
        nshapes[prx(idxs[-1])] = tuple(shmap[x] for x in script.split('->')[1])
    return nshapes, nwork, nflops

def ut_dfs_sum_lift_evaluate(iout, lf_eqss, n_occ, n_virt, n_cas, ts, ints, use_hast, out_work=None, work=None, use_tbl=False):
    import numpy as np
    from functools import reduce
    prx = lambda idx: (idx.count('I') + idx.count('A') + idx.count('E')) // 2 * (1 + idx.count('G'))
    iprx = lambda idx: ((len(idx) + 1) // 2) * ((len(idx) + 1) // 2 + 1) // 2 - 1 + prx(idx)
    sli = {'i': slice(n_occ[0]), 'a': slice(n_occ[0], n_occ[0] + n_cas[0]), 'e': slice(n_occ[0] + n_cas[0], None),
        'I': slice(n_occ[1]), 'A': slice(n_occ[1], n_occ[1] + n_cas[1]), 'E': slice(n_occ[1] + n_cas[1], None), 'G': slice(None)}
    slish = {'i': n_occ[0], 'a': n_cas[0], 'e': n_virt[0], 'I': n_occ[1], 'A': n_cas[1], 'E': n_virt[1]}
    if ints is not None and len(ints) > 2 and list(ints[2].values())[0].ndim == 3:
        slish['G'] = list(ints[2].values())[0].shape[0]
    r = [None if use_hast else 0.0] * (max([0] + [prx(x[2][-1]) for x in lf_eqss[iout]]) + 1)
    iow = 0
    for ix, (f, script, idxs, nm, *_) in enumerate(lf_eqss[iout]):
        iwork = 0
        tensors = []
        for nmx, idx in zip(nm[:-1], idxs[:-1]):
            xshape = tuple(slish[x] for x in idx)
            xlen = reduce(lambda x, y: x * y, xshape, 1)
            if nmx[0] == '_':
                if work is not None:
                    xout_work, xwork = work[iwork:iwork + xlen], work[iwork + xlen:]
                    iwork += xlen
                else:
                    xout_work, xwork = None, None
                tensors.append(ut_dfs_sum_lift_evaluate(int(nmx[nmx.count('_'):]), lf_eqss,
                    n_occ, n_virt, n_cas, ts, ints, use_hast, out_work=xout_work, work=xwork)[prx(idx)])
            else:
                if nmx[0] in 'LRT':
                    tensors.append(ts[nmx][(len(idx) + 1) // 2 - 1][prx(idx)])
                elif isinstance(ints[iprx(idx)], dict):
                    tensors.append(ints[iprx(idx)][idx])
                else:
                    tensors.append(ints[iprx(idx)][tuple(sli[x] for x in idx)])
                if use_hast:
                    from pyhast.tensor.packed import PackedTensor
                    tensors[-1] = PackedTensor(tensors[-1])
        if not use_hast:
            r[prx(idxs[-1])] += f * np.einsum(script, *tensors, optimize='optimal')
        else:
            from pyhast.tensor.packed import einsum, PackedTensor
            if r[prx(idxs[-1])] is None:
                shmap = {k: s for ten, scr in zip(tensors, script.split('->')[0].split(',')) for k, s in zip(scr, ten.shape)}
                nshape = tuple(shmap[x] for x in script.split('->')[1])
                if out_work is not None:
                    r[prx(idxs[-1])] = PackedTensor(out_work[iow:iow + reduce(lambda x, y: x * y, nshape, 1)], shape=nshape)
                    iow += reduce(lambda x, y: x * y, nshape, 1)
                else:
                    r[prx(idxs[-1])] = PackedTensor.empty(nshape)
                einsum(script, *tensors, alpha=f, beta=0, out=r[prx(idxs[-1])], use_tbl=use_tbl)
            else:
                einsum(script, *tensors, alpha=f, beta=1, out=r[prx(idxs[-1])], use_tbl=use_tbl)
    return r

def ut_evaluate(eqs, n_occ, n_virt, n_cas, ts, ints, eval_t=None, with_blocks=False):
    import numpy as np
    if len(eqs) == 0:
        return []
    elif len(eqs) == 1 and len(eqs[0]) == 0:
        return [[0.0]] if not with_blocks else [{}]
    prx = lambda idx: (idx.count('I') + idx.count('A') + idx.count('E')) // 2 * (1 + idx.count('G'))
    iprx = lambda idx: ((len(idx) + 1) // 2) * ((len(idx) + 1) // 2 + 1) // 2 - 1 + prx(idx)
    ts_new = [[0.0] * (it + 2) for it in range(len(eqs))] if not with_blocks else [{} for _ in eqs]
    ats = {'&': {'ee': np.identity(n_virt[0]), 'ii': np.identity(n_occ[0]), 'aa': np.identity(n_cas[0]),
        'EE': np.identity(n_virt[1]), 'II': np.identity(n_occ[1]), 'AA': np.identity(n_cas[1])},
        '=': {'e': np.ones(n_virt[0]), 'i': np.ones(n_occ[0]), 'a': np.ones(n_cas[0]),
        'E': np.ones(n_virt[1]), 'I': np.ones(n_occ[1]), 'A': np.ones(n_cas[1])}}
    feqs = [xeq for xeq in eqs if len(xeq) != 0]
    try:
        import opt_einsum
        xcontract = opt_einsum.contract
    except ImportError:
        xcontract = lambda script, *tensors: np.einsum(script, *tensors, optimize='optimal')
    if len(feqs) == 0:
        return ts_new
    if isinstance(feqs[0][0], int) and eval_t == 'dfs':
        nouts, lf_eqss = eqs[0][:2]
        ts_new = [ut_dfs_sum_lift_evaluate(iout, lf_eqss, n_occ, n_virt, n_cas, ts, ints, False, use_tbl=False) for iout in range(-nouts, 0)]
    elif isinstance(feqs[0][0], int) and eval_t in ['hast-dfs', 'tblis-dfs']:
        nouts, lf_eqss = eqs[0][:2]
        ts_new = [[x.to_array() for x in ut_dfs_sum_lift_evaluate(iout, lf_eqss, n_occ, n_virt, n_cas, ts, ints, True, use_tbl='tblis' in eval_t)] for iout in range(-nouts, 0)]
    elif isinstance(feqs[0][0], int) and eval_t in ['hast-work-dfs', 'tblis-work-dfs']:
        from functools import reduce
        nouts, lf_eqss = eqs[0][:2]
        ts_new = []
        for iout in range(-nouts, 0):
            nshapes, nwork, _ = ut_dfs_sum_lift_work_size(iout, lf_eqss, n_occ, n_virt, n_cas)
            nout_works = [reduce(lambda x, y: x * y, nshape, 1) for nshape in nshapes]
            out_work, work = np.empty((sum(nout_works), )), np.empty((nwork, ))
            ts_new.append([x.to_array() for x in ut_dfs_sum_lift_evaluate(iout, lf_eqss, n_occ, n_virt, n_cas, ts, ints, True, out_work=out_work, work=work, use_tbl='tblis' in eval_t)])
    elif isinstance(feqs[0][0], int) and eval_t is None:
        nouts, lf_eqss, imdels = eqs[0]
        ims = []
        for iif, (lf_eqs, imds) in enumerate(zip(lf_eqss, imdels)):
            if with_blocks:
                r = {}
            elif iif >= len(lf_eqss) - nouts:
                r = [0.0] * (iif - (len(lf_eqss) - nouts) + 2)
            else:
                r = 0
            for f, script, idxs, nm, *_ in lf_eqs:
                tensors = []
                for nmx, idx in zip(nm[:-1], idxs[:-1]):
                    tensors.append(ims[int(nmx[nmx.count('_'):])] if nmx[0] == '_' else (
                        ts[nmx][(len(idx) + 1) // 2 - 1][prx(idx)] if nmx[0] in 'LRT#^' else
                        (ats[nmx][idx] if nmx in '&=' else ints[iprx(idx)][idx])))
                    if isinstance(tensors[-1], dict):
                        tensors[-1] = tensors[-1][idx]
                if with_blocks:
                    if idxs[-1] in r:
                        r[idxs[-1]] += f * xcontract(script, *tensors)
                    else:
                        r[idxs[-1]] = f * xcontract(script, *tensors)
                elif iif >= len(lf_eqss) - nouts:
                    r[prx(idxs[-1])] += f * xcontract(script, *tensors)
                else:
                    r += f * xcontract(script, *tensors)
            for im in imds:
                ims[im] = None
            ims.append(r)
        ts_new = ims[-nouts:]
    elif len(feqs[0][0]) == 4:
        for kk, xeqs in enumerate(eqs):
            for f, script, idxs, nm in xeqs:
                tensors = []
                for nmx, idx in zip(nm[:-1], idxs[:-1]):
                    tensors.append(ts[nmx][(len(idx) + 1) // 2 - 1][prx(idx)] if nmx[0] in 'LRT#^' else
                        (ats[nmx][idx] if nmx in '&=' else ints[iprx(idx)][idx]))
                    if isinstance(tensors[-1], dict):
                        tensors[-1] = tensors[-1][idx]
                if with_blocks:
                    if idxs[-1] in ts_new[kk]:
                        ts_new[kk][idxs[-1]] += f * xcontract(script, *tensors)
                    else:
                        ts_new[kk][idxs[-1]] = f * xcontract(script, *tensors)
                else:
                    ts_new[kk][prx(idxs[-1])] += f * xcontract(script, *tensors)
    elif len(feqs[0][0]) in [2, 3] and isinstance(feqs[0][0][1], list):
        for kk, xeqs in enumerate(eqs):
            for f, ex_eqs in [x[:2] for x in xeqs]:
                ims = {}
                for assoc, script, idxs, nm in ex_eqs:
                    tensors = []
                    for nmx, idx in zip(nm[:-1], idxs[:-1]):
                        tensors.append(ims[nmx] if isinstance(nmx, tuple) else (
                            ts[nmx][(len(idx) + 1) // 2 - 1][prx(idx)] if nmx[0] in 'LRT#^' else
                            (ats[nmx][idx] if nmx in '&=' else ints[iprx(idx)][idx])))
                        if isinstance(tensors[-1], dict):
                            tensors[-1] = tensors[-1][idx]
                    ims[assoc] = xcontract(script, *tensors)
                if with_blocks:
                    if ex_eqs[-1][2][-1] in ts_new[kk]:
                        ts_new[kk][ex_eqs[-1][2][-1]] += f * ims[ex_eqs[-1][0]]
                    else:
                        ts_new[kk][ex_eqs[-1][2][-1]] = f * ims[ex_eqs[-1][0]]
                else:
                    ts_new[kk][prx(ex_eqs[-1][2][-1])] += f * ims[ex_eqs[-1][0]]
    else:
        assert False
    return ts_new

def ut_symm_dfs_sum_lift_summary(lf_eqsss, n_occ, n_virt, n_cas):
    from pyhast.sr.utils import format_size
    from functools import reduce
    print("%5s %50s %10s %10s %14s" % ("order", "shape", "amp size", "work size", "cost"))
    for p, xeqs in enumerate(lf_eqsss):
        nouts, lf_eqss = xeqs[:2]
        for iout in range(-nouts, 0):
            shapes, sz, fl = ut_symm_dfs_sum_lift_work_size(iout, lf_eqss, n_occ, n_virt, n_cas)
            szt = sum([reduce(lambda x, y: x * y, shape, 1) for shape in shapes])
            print("%5d %50s %10s %10s %14s" % (iout + nouts + p, shapes[0], format_size(szt * 8),
                format_size(sz * 8), format_size(fl, 'FLOPs')))

def ut_symm_dfs_sum_lift_work_size(iout, lf_eqss, n_occ, n_virt, n_cas):
    from pyhast.tensor.hastctr import symm_contract_nflops, symm_tensor_shape
    from functools import reduce
    prx = lambda idx: (idx.count('I') + idx.count('A') + idx.count('E')) // 2
    nwork, nflops = 0, 0
    nshapes = [()] * (max([0] + [prx(x[2][-1]) for x in lf_eqss[iout]]) + 1)
    for _, script, idxs, nm, xsymms in lf_eqss[iout]:
        tensors = []
        ztensor = 0
        for nmx, idx, xsm in zip(nm[:-1], idxs[:-1], xsymms[:-1]):
            if nmx[0] == '_':
                xshapes, xwork, xflops = ut_symm_dfs_sum_lift_work_size(int(nmx[nmx.count('_'):]), lf_eqss, n_occ, n_virt, n_cas)
                tensors.append(xshapes[prx(idx)])
                ztensor += reduce(lambda x, y: x * y, xshapes[prx(idx)], 1)
                nwork = max(nwork, xwork + ztensor)
                nflops += xflops
            else:
                tensors.append(symm_tensor_shape(n_occ, n_virt, n_cas, len(idx), xsm))
        nflops += symm_contract_nflops(n_occ, n_virt, n_cas, script, xsymms)
        nshapes[prx(idxs[-1])] = symm_tensor_shape(n_occ, n_virt, n_cas, len(idxs[-1]), xsymms[-1])
    return nshapes, nwork, nflops

def ut_symm_dfs_sum_lift_evaluate(iout, lf_eqss, n_occ, n_virt, n_cas, ts, ints, out_work=None, work=None):
    import numpy as np
    from pyhast.tensor.hastctr import symm_tensor_shape, symm_contract
    from functools import reduce
    prx = lambda idx: (idx.count('I') + idx.count('A') + idx.count('E')) // 2 * (1 + idx.count('G'))
    iprx = lambda idx: ((len(idx) + 1) // 2) * ((len(idx) + 1) // 2 + 1) // 2 - 1 + prx(idx)
    r = [None] * (max([0] + [prx(x[2][-1]) for x in lf_eqss[iout]]) + 1)
    iow = 0
    for ix, (f, script, idxs, nm, xsymms) in enumerate(lf_eqss[iout]):
        iwork = 0
        tensors = []
        for nmx, idx, xsm in zip(nm[:-1], idxs[:-1], xsymms[:-1]):
            xshape = symm_tensor_shape(n_occ, n_virt, n_cas, len(idx), xsm)
            xlen = reduce(lambda x, y: x * y, xshape, 1)
            if nmx[0] == '_':
                if work is not None:
                    xout_work, xwork = work[iwork:iwork + xlen], work[iwork + xlen:]
                    iwork += xlen
                else:
                    xout_work, xwork = None, None
                tensors.append(ut_symm_dfs_sum_lift_evaluate(int(nmx[nmx.count('_'):]), lf_eqss,
                    n_occ, n_virt, n_cas, ts, ints, out_work=xout_work, work=xwork)[prx(idx)])
            else:
                if nmx[0] in 'LRT':
                    tensors.append(ts[nmx][(len(idx) + 1) // 2 - 1][prx(idx)])
                else:
                    tensors.append(ints[iprx(idx)][idx])
        if r[prx(idxs[-1])] is None:
            nshape = symm_tensor_shape(n_occ, n_virt, n_cas, len(idxs[-1]), xsymms[-1])
            if out_work is not None:
                r[prx(idxs[-1])] = out_work[iow:iow + reduce(lambda x, y: x * y, nshape, 1)].reshape(nshape)
                iow += reduce(lambda x, y: x * y, nshape, 1)
            else:
                r[prx(idxs[-1])] = np.empty(nshape)
            symm_contract(n_occ, n_virt, n_cas, script, xsymms, *tensors, alpha=f, beta=0, out=r[prx(idxs[-1])])
        else:
            symm_contract(n_occ, n_virt, n_cas, script, xsymms, *tensors, alpha=f, beta=1, out=r[prx(idxs[-1])])
    return r

def ut_symm_evaluate(eqs, n_occ, n_virt, n_cas, ts, ints, merge_schemes, eval_t=None, with_blocks=False):
    import pyhast.wick.gsymm, numpy as np
    prx = lambda idx: (idx.count('I') + idx.count('A') + idx.count('E')) // 2 * (1 + idx.count('G'))
    iprx = lambda idx: ((len(idx) + 1) // 2) * ((len(idx) + 1) // 2 + 1) // 2 - 1 + prx(idx)
    ts_new = [[0.0] * (it + 2) for it in range(len(eqs))] if not with_blocks else [{} for _ in eqs]
    ats = {'&': {'ee': np.identity(n_virt[0]), 'ii': np.identity(n_occ[0]), 'aa': np.identity(n_cas[0]),
        'EE': np.identity(n_virt[1]), 'II': np.identity(n_occ[1]), 'AA': np.identity(n_cas[1])},
        '=': {'e': np.ones(n_virt[0]), 'i': np.ones(n_occ[0]), 'a': np.ones(n_cas[0]),
        'E': np.ones(n_virt[1]), 'I': np.ones(n_occ[1]), 'A': np.ones(n_cas[1])}}
    feqs = [xeq for xeq in eqs if len(xeq) != 0]
    xcontract = lambda *args: pyhast.wick.gsymm.gt_symm_contract(merge_schemes, *args)
    if len(feqs) == 0:
        return ts_new
    if isinstance(feqs[0][0], int) and eval_t == 'hastctr-dfs':
        nouts, lf_eqss = eqs[0][:2]
        ts_new = [ut_symm_dfs_sum_lift_evaluate(iout, lf_eqss, n_occ, n_virt, n_cas, ts, ints) for iout in range(-nouts, 0)]
    elif isinstance(feqs[0][0], int) and eval_t == 'hastctr-work-dfs':
        from functools import reduce
        nouts, lf_eqss = eqs[0][:2]
        ts_new = []
        for iout in range(-nouts, 0):
            nshapes, nwork, _ = ut_symm_dfs_sum_lift_work_size(iout, lf_eqss, n_occ, n_virt, n_cas)
            nout_works = [reduce(lambda x, y: x * y, nshape, 1) for nshape in nshapes]
            out_work, work = np.empty((sum(nout_works), )), np.empty((nwork, ))
            ts_new.append(ut_symm_dfs_sum_lift_evaluate(iout, lf_eqss, n_occ, n_virt, n_cas, ts, ints, out_work=out_work, work=work))
    elif isinstance(feqs[0][0], int) and eval_t in [None, 'hastctr']:
        nouts, lf_eqss, imdels = eqs[0]
        ims = []
        for iif, (lf_eqs, imds) in enumerate(zip(lf_eqss, imdels)):
            r = None if eval_t == 'hastctr' else 0
            if iif >= len(lf_eqss) - nouts:
                r = [r] * (iif - (len(lf_eqss) - nouts) + 2)
            if with_blocks:
                r = {}
            for f, script, idxs, nm, xsymms in lf_eqs:
                tensors = []
                for nmx, idx in zip(nm[:-1], idxs[:-1]):
                    tensors.append(ims[int(nmx[nmx.count('_'):])] if nmx[0] == '_' else (
                        ts[nmx][(len(idx) + 1) // 2 - 1][prx(idx)] if nmx[0] in 'LRT#^' else
                        (ats[nmx][idx] if nmx in '&=' else ints[iprx(idx)][idx])))
                    if isinstance(tensors[-1], dict):
                        tensors[-1] = tensors[-1][idx]
                if eval_t == 'hastctr':
                    if with_blocks:
                        if idxs[-1] in r:
                            r[idxs[-1]] = pyhast.tensor.hastctr.symm_contract(n_occ, n_virt, n_cas, script, xsymms,
                                *tensors, alpha=f, beta=1.0, out=r[idxs[-1]])
                        else:
                            r[idxs[-1]] = pyhast.tensor.hastctr.symm_contract(n_occ, n_virt, n_cas, script, xsymms,
                                *tensors, alpha=f, beta=0.0, out=None)
                    elif iif >= len(lf_eqss) - nouts:
                        beta = 0.0 if r[prx(idxs[-1])] is None else 1.0
                        r[prx(idxs[-1])] = pyhast.tensor.hastctr.symm_contract(n_occ, n_virt, n_cas, script, xsymms,
                            *tensors, alpha=f, beta=beta, out=r[prx(idxs[-1])])
                    else:
                        beta = 0.0 if r is None else 1.0
                        r = pyhast.tensor.hastctr.symm_contract(n_occ, n_virt, n_cas, script, xsymms,
                            *tensors, alpha=f, beta=beta, out=r)
                else:
                    if with_blocks:
                        if idxs[-1] in r:
                            r[idxs[-1]] += f * xcontract(script, xsymms, *tensors)
                        else:
                            r[idxs[-1]] = f * xcontract(script, xsymms, *tensors)
                    elif iif >= len(lf_eqss) - nouts:
                        r[prx(idxs[-1])] += f * xcontract(script, xsymms, *tensors)
                    else:
                        r += f * xcontract(script, xsymms, *tensors)
            for im in imds:
                ims[im] = None
            ims.append([rr if rr is not None else 0.0 for rr in r] if isinstance(r, list) else (r if r is not None else 0.0))
        ts_new = ims[-nouts:]
    elif len(feqs[0][0]) in [2, 3] and isinstance(feqs[0][0][1], list):
        for kk, xeqs in enumerate(eqs):
            for f, ex_eqs, symms in xeqs:
                ims = {}
                assert len(ex_eqs) == len(symms)
                for (assoc, script, idxs, nm), xsymms in zip(ex_eqs, symms):
                    tensors = []
                    for nmx, idx in zip(nm[:-1], idxs[:-1]):
                        tensors.append(ims[nmx] if isinstance(nmx, tuple) else (
                            ts[nmx][(len(idx) + 1) // 2 - 1][prx(idx)] if nmx[0] in 'LRT#^' else
                            (ats[nmx][idx] if nmx in '&=' else ints[iprx(idx)][idx])))
                    if eval_t is None:
                        ims[assoc] = pyhast.wick.gsymm.gt_symm_contract(merge_schemes, script, xsymms, *tensors)
                    elif eval_t == 'hastctr':
                        ims[assoc] = pyhast.tensor.hastctr.symm_contract(n_occ, n_virt, n_cas, script, xsymms, *tensors)
                if with_blocks:
                    if ex_eqs[-1][2][-1] in ts_new[kk]:
                        ts_new[kk][ex_eqs[-1][2][-1]] += f * ims[ex_eqs[-1][0]]
                    else:
                        ts_new[kk][ex_eqs[-1][2][-1]] = f * ims[ex_eqs[-1][0]]
                else:
                    if ims[ex_eqs[-1][0]].size == 0:
                        ts_new[kk][prx(ex_eqs[-1][2][-1])] = ims[ex_eqs[-1][0]]
                    else:
                        ts_new[kk][prx(ex_eqs[-1][2][-1])] += f * ims[ex_eqs[-1][0]]
    else:
        assert False
    return ts_new
