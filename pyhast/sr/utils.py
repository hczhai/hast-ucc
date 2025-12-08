
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

def cc_run_diis(cc, tamps, istep, normt, de, adiis):
    from pyscf.lib import logger
    if adiis and istep >= cc.diis_start_cycle:
        tamps, diis_order = list(tamps), getattr(cc, 'diis_order', len(tamps))
        diis_order = len(tamps) if diis_order is None else diis_order
        if cc.diis_symm_amps:
            vec = cc.amplitudes_to_symm_vector(tamps[:diis_order])
            tamps[:diis_order] = cc.symm_vector_to_amplitudes(adiis.update(vec), diis_order)
        else:
            vec = cc.amplitudes_to_vector(tamps[:diis_order])
            tamps[:diis_order] = cc.vector_to_amplitudes(adiis.update(vec), diis_order)
        logger.debug1(cc, "DIIS for step %d", istep)
    return tamps

def format_size(i, suffix='B'):
    if i < 1000:
        return "%d %s" % (i, suffix)
    else:
        a = 1024
        for pf in "KMGTPEZY":
            p = 2
            for k in [10, 100, 1000]:
                if i < k * a:
                    return "%%.%df %%s%%s" % p % (i / a, pf, suffix)
                p -= 1
            a *= 1024
    return "??? " + suffix

def get_peak_memory():
    import resource, sys
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    return rusage.ru_maxrss * (1 if sys.platform == 'darwin' else 1024)

def get_diff_norm(xtamps, ytamps):
    import numpy as np
    if isinstance(xtamps, (tuple, list)):
        return np.linalg.norm([get_diff_norm(x, y) for x, y in zip(xtamps, ytamps)])
    elif isinstance(xtamps, dict):
        return np.linalg.norm([get_diff_norm(xtamps[k], ytamps[k]) for k in set(list(xtamps) + list(ytamps))])
    xtamps, ytamps = xtamps.ravel(), ytamps.ravel()
    r, nb = 0.0, 24 * 1024 * 1024
    for i in range(0, len(xtamps), nb):
        r += np.linalg.norm(xtamps[i:i + nb] - ytamps[i:i + nb]) ** 2
    return r ** 0.5

def cc_kernel(cc, tamps=None, max_cycle=50, tol=1e-8, tolnormt=1e-6, lamps=None, xlamps=None, xramps=None,
              lamb=False):
    from pyscf.lib import logger
    import numpy as np, time
    from pyhast.sr.diis import DIIS

    log = logger.new_logger(cc, cc.verbose)
    if tamps is None:
        tamps = cc.tamps if lamb else cc.get_init_guess()
    if lamps is None and lamb:
        lamps = cc.zero_amps(cc.lamb_order)
        lamps[:len(cc.tamps)] = [list(x) if isinstance(x, list) else x for x in cc.tamps]
    xamps = [tamps, lamps][lamb]
    lamps = None

    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    t0 = time.perf_counter()
    eold, erefold = 0, 0
    if hasattr(cc, 'ncas'):
        log.info("Nocc = %r Nact = %r Nvir = %r Amps size = %s", cc.ncore, cc.ncas, cc.nvirt,
            format_size(cc.amplitudes_to_vector(xamps).nbytes))
    else:
        log.info("Nocc = %r Nvir = %r Amps size = %s", cc.n_occ, cc.n_virt,
            format_size(cc.amplitudes_to_vector(xamps).nbytes))
    if hasattr(cc, 'update_effective_ints'):
        cc.update_effective_ints(tamps)
    ecc = cc.energy(tamps) if not lamb else cc.e_corr
    log.info("Init E_corr(%s) = %.15g M = %8s", cc.name, ecc, format_size(get_peak_memory()))

    conv = False
    cc_name = cc.name

    if cc.newton_krylov:
        import scipy
        try:
            def f(x, istep=np.array(0)):
                if getattr(cc, 'amps_save_dir', None) is not None:
                    np.save(cc.amps_save_dir + "/%s-%s.npy" % (cc.name, "lamps" if lamb else "tamps"), x)
                t1 = time.perf_counter()
                xamps = cc.vector_to_amplitudes(x, cc.order)
                if hasattr(cc, 'update_effective_ints'):
                    cc.update_effective_ints(xamps)
                xamps_new = (cc.update_lambda(xamps, tamps) if lamb else cc.update_amps(xamps))
                newx = cc.amplitudes_to_vector(xamps_new)
                normt = np.linalg.norm(newx)
                peak_mem = format_size(get_peak_memory())
                if not lamb:
                    log.info("cycle =%3d  E_corr(%s) = %22.15f  norm(r tamps) = %10.3E M = %8s T = %12.3f",
                        istep + 1, cc.name, cc.energy(xamps), normt, peak_mem, time.perf_counter() - t1)
                else:
                    log.info("cycle =%3d  Lambda(%s)  norm(r lamps) = %10.3E M = %8s T = %12.3f",
                        istep + 1, cc_name, normt, peak_mem, time.perf_counter() - t1)
                istep += 1
                return newx
            x = cc.amplitudes_to_vector(xamps)
            x = scipy.optimize.newton_krylov(f, x, maxiter=max_cycle, method='lgmres', f_tol=tolnormt * 0.5,
                tol_norm=np.linalg.norm, line_search='wolfe', rdiff=1E-6, verbose=cc.verbose >= 4)
            conv, xamps = True, cc.vector_to_amplitudes(x, cc.order)
        except scipy.optimize.NoConvergence as nconv:
            conv, xamps = False, cc.vector_to_amplitudes(nconv.args[0], cc.order)
        if not lamb and hasattr(cc, 'update_effective_ints'):
            cc.update_effective_ints(xamps)
        ecc = cc.energy(xamps) if not lamb else cc.e_corr
        max_cycle = 0

    if max_cycle != 0 and cc.diis:
        adiis = DIIS(cc, space=cc.diis_space, scratch=getattr(cc, 'diis_scratch', None),
            scratch_start=getattr(cc, 'diis_scratch_start', 0))
    else:
        adiis = None

    for istep in range(max_cycle):
        t1 = time.perf_counter()
        xamps_new = (cc.update_lambda(xamps, tamps) if lamb else cc.update_amps(xamps))
        normt = get_diff_norm(xamps_new, xamps)
        if cc.iterative_damping < 1.0:
            alpha = cc.iterative_damping
            for tx, tx_new in zip(xamps, xamps_new):
                tx_new *= alpha
                tx_new += (1 - alpha) * tx
        xamps = xamps_new
        xamps_new = None
        xamps = cc.run_diis(xamps, istep, normt, ecc - eold, adiis)
        if getattr(cc, 'amps_save_dir', None) is not None:
            np.save(cc.amps_save_dir + "/%s-%s.npy" % (cc.name, "lamps" if lamb else "tamps"), cc.amplitudes_to_vector(xamps))
        if hasattr(cc, 'update_effective_ints'):
            cc.update_effective_ints(xamps)
        eold, ecc = ecc, cc.energy(xamps) if not lamb else cc.e_corr
        peak_mem = format_size(get_peak_memory())
        if not lamb:
            log.info("cycle =%3d  E_corr(%s) = %22.15f  dE = %10.3E  norm(d tamps) = %10.3E M = %8s T = %12.3f",
                istep + 1, cc.name, ecc, ecc - eold, normt, peak_mem, time.perf_counter() - t1)
        else:
            log.info("cycle =%3d  Lambda(%s)  norm(d lamps) = %10.3E M = %8s T = %12.3f",
                istep + 1, cc_name, normt, peak_mem, time.perf_counter() - t1)
        cput1 = log.timer("%s iter" % cc.name, *cput1)
        if abs(ecc - eold) < tol and normt < tolnormt:
            conv = True
            break
    log.timer(cc.name, *cput0)
    logger.info(cc, "%s %sconverged", cc_name, ("lambda " if lamb else "") + ("" if conv else "not "))
    if lamb:
        cc.lamb_converged, cc.lamps = conv, xamps
    else:
        cc.converged, cc.e_corr, cc.tamps = conv, ecc, xamps
        logger.note(cc, "E(%s) = %.16g  E_corr = %.16g", cc.name, cc.e_tot, cc.e_corr)
    logger.note(cc, "Time = %12.3f", time.perf_counter() - t0)

    if adiis is not None:
        adiis.clean_scratch()

    return cc.e_tot

def cc_lamb_kernel(cc, tamps=None, lamps=None, **kwargs):
    cc_kernel(cc, tamps, lamps=lamps, lamb=True, **kwargs)
    return cc.lamb_converged, cc.lamps

def cc_npdm_kernel(cc, order, tamps=None, lamps=None, unpack=True):
    tamps = cc.tamps if tamps is None else tamps
    lamps = cc.lamps if lamps is None else lamps
    return cc.npdm_intermediates(order, tamps, lamps, unpack=unpack)

def cc_prim_eqs(t_ord, l_ord=None, p_ord=None, max_comm=None, norm=False, normal_ord=True):
    from pyhast.wick.gwick import gt_initialize, is_conn
    if p_ord is not None and l_ord is not None:
        prefs, names, (pi, pj), prim_eqs = [1], ['LX'], (None, 0), []
        inits = [[[l, x] for x in range(2, p_ord * 2 + 1, 2) for l in range(0, l_ord * 2 + 1, 2)]]
    elif l_ord is None:
        prefs, names, (pi, pj), prim_eqs = [1], ['XH'], (None, 0), []
        inits = [[[x, h] for x in range(0, t_ord * 2 + 1, 2) for h in [2, 4]]]
    else:
        prefs, names, (pi, pj), prim_eqs = [1], ['LHX'], (-1, None), []
        inits = [[[l, h, x] for x in range(2, l_ord * 2 + 1, 2) for h in [0, 2, 4] for l in range(0, l_ord * 2 + 1, 2)]]
    for it in range(0, min([4 if p_ord is None else p_ord * 2] + [x for x in [max_comm] if x is not None] + [[], [0]][norm]) + 1):
        for pf, nms, xinit in zip(prefs, names, inits):
            conn_nms = ['HGT', 'XT'][p_ord is not None]
            eqs, nctrs, pts = [], [], [i for i, nm in enumerate(nms) if nm in conn_nms]
            ipts = [i for i, nm in enumerate(nms) if nm in ['HG', 'X'][p_ord is not None]]
            for init in sorted(xinit, key=lambda x: x[::-1]):
                ctr_idxs = [(j, i) for i in range(len(init)) for j in range(i) if nms[j] + nms[i] not in ['TT', 'TX', 'TR', 'GG', 'RX']]
                ctr_idxs = ctr_idxs if normal_ord else ctr_idxs \
                    + [(j, i) for i in range(len(init)) for j in range(i + 1) if nms[i] + nms[j] in ['HH', 'GG']]
                init_eqs = [eq for eq in gt_initialize(init, ctr_idxs) if is_conn(eq, pts, inits=ipts) or norm]
                if len(init_eqs) != 0:
                    eqs.append(init_eqs)
                    nctrs.append(sum(init) // 2)
            if len(nctrs) != 0:
                prim_eqs.append((pf, eqs, nctrs, len(nms), list(nms)))
        if it != min([4 if p_ord is None else p_ord * 2] + [x for x in [max_comm] if x is not None]):
            inits = [[init[:pi] + [t] + init[pi:pj] for init in xinit for t in range(2, t_ord * 2 + 1, 2)] for xinit in inits]
            names = [nms[:pi] + 'T' + nms[pi:pj] for nms in names]
            prefs = [pf / (it + 1) for pf in prefs]
    return prim_eqs
