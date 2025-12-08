
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

import pyhast.wick.uwick
from pyhast.sr.utils import cc_prim_eqs
from pyhast.sr.utils import cc_kernel, cc_run_diis, cc_lamb_kernel, cc_npdm_kernel
from pyhast.sr.ao2mo import ut_ao2mo

def ucc_energy(cc, tamps):
    ints, n_occ, n_virt, eqs = cc.ints, cc.n_occ, cc.n_virt, cc.eqs
    ener = pyhast.wick.uwick.ut_symm_evaluate(eqs[:1], n_occ, n_virt, (0, 0), {'T': tamps}, ints,
        cc.merge_schemes, eval_t=cc.eval_t)[0][0]
    return ener

def ucc_update_amps(cc, tamps, lamps=[]):
    import numpy as np, math
    do_lamb = len(lamps) != 0
    ints, (nocca, noccb), (nvira, nvirb), eqs = cc.ints, cc.n_occ, cc.n_virt, [cc.eqs, cc.lamb_eqs][do_lamb]
    ints = ints + [{'': np.array(-cc.e_corr) if do_lamb else 0.0}] * 2
    lamps = list(lamps) + [[np.array(1.0)]]
    ts_new = pyhast.wick.uwick.ut_symm_evaluate(eqs[1:], (nocca, noccb), (nvira, nvirb), (0, 0),
        {'T': tamps, 'L': lamps}, ints, cc.merge_schemes, eval_t=cc.eval_t)
    for kk, k in [(kk, k) for kk in range(len(ts_new)) for k in range(len(tamps[kk]))]:
        ts_new[kk][k] *= (math.factorial(k) * math.factorial(kk + 1 - k)) ** 2
    if not cc.newton_krylov:
        for kk in range(len(ts_new)):
            ts_new[kk] = ts_new[kk][:len(tamps[kk])]
            for k in range(len(ts_new[kk])):
                exxa = cc.exxs[kk + 1 - k][0][:, None] - cc.exxs[kk + 1 - k][1][None, :]
                exxb = cc.exxs[k][2][:, None] - cc.exxs[k][3][None, :]
                exx = exxa[:, None, :, None] + exxb[None, :, None] if k != 0 and k != kk + 1 else [exxa, exxb][k != 0]
                xt = [tamps, lamps][do_lamb][kk][k]
                ts_new[kk][k] = (ts_new[kk][k] + exx * xt) / exx
    return ts_new

def ucc_npdm_intermediates(cc, order, tamps, lamps=None, xlamps=None, xramps=None, ipea=0, unpack=True):
    import numpy as np, pyhast.wick.usymm
    (nocca, noccb), (nvira, nvirb), dms = cc.n_occ, cc.n_virt, [{} for _ in range(order)]
    if lamps is not None:
        lamps = list(lamps) + [[np.array(0.0 if xlamps is not None and xramps is not None else 1.0)]]
        assert cc.lift or order < len(cc.npdm_eqs)
        xdms = pyhast.wick.uwick.ut_symm_evaluate(cc.npdm_eqs[:order + 1], (nocca, noccb), (nvira, nvirb), (0, 0),
            {'T': tamps, 'L': lamps}, None, cc.merge_schemes, eval_t=cc.eval_t, with_blocks=True)[1:]
        xdms = [{k: pyhast.wick.usymm.ut_unsymm_npdm(v, k, cc.symm_schemes) for k, v in xdm.items()} for xdm in xdms]
        for dm, k, v in [(dm, k, v) for dm, xdm in zip(dms, xdms) for k, v in xdm.items()]:
            dm[k] = dm.get(k, 0.0) + v

    if unpack:
        sli = {'i': slice(nocca), 'e': slice(nocca, None), 'I': slice(noccb), 'E': slice(noccb, None)}
        for idm, dm in enumerate(dms):
            tags = ['i' * (idm + 1 - ik) + 'I' * ik for ik in range(idm + 2)]
            upk_dms = [np.zeros((nocca + nvira, ) * ((idm + 1) * 2), dtype=tamps[0][0].dtype) for _ in range(idm + 2)]
            for k, v in dm.items():
                upk_dms[(k.count('I') + k.count('E')) // 2][tuple(sli[x] for x in k)] = v
            dms[idm] = [pyhast.wick.gwick.gt_npdm_symmetrize(pyhast.wick.uwick.ut_antisymmetrize(x, tag * 2))
                for x, tag in zip(upk_dms, tags)]
        dms = [[np.pad(xdm, (cc.nfrozen, 0)) for xdm in dm] for dm in dms] if cc.nfrozen != 0 else dms
        dms = pyhast.wick.uwick.ut_npdm_denormal_order(dms, (nocca + cc.nfrozen, noccb + cc.nfrozen))
        for idm, dm in enumerate(dms):
            dms[idm] = [xdm.transpose(tuple(j for i in range(idm + 1) for j in [i, i + idm + 1])) for xdm in dm]
    return dms

def ucc_amplitudes_to_vector(cc, tamps, out=None):
    import numpy as np
    vector = np.ndarray(np.sum([tt.size for t in tamps for tt in t]), tamps[0][0].dtype, buffer=out)
    size = 0
    for tt in [tt for t in tamps for tt in t]:
        vector[size : size + tt.size] = tt.ravel()
        size += tt.size
    return vector

def ucc_vector_to_amplitudes(cc, vector, order, has_ref=False, ipea=0):
    import numpy as np
    z, it, tamps = 0, max(abs(ipea) - 1, 0) + (-1 if has_ref else 0), []
    for it in range(it, order):
        t = []
        for k, kk in [(k, kk) for k in range(it + 2 - abs(ipea)) for kk in range(abs(ipea) + 1)]:
            xx = (it + 1 - k - max(ipea, 0) - kk * (ipea < 0), k + kk * (ipea < 0),
                it + 1 - k + min(ipea, 0) - kk * (ipea > 0), k + kk * (ipea > 0))
            x = tuple(cc.symm_schemes[0][p][q] for p, q in zip(xx, [0, 2, 1, 3]) if p)
            t.append(vector[z:z + np.prod(x, dtype=int)].reshape(x))
            z += np.prod(x, dtype=int)
        tamps.append(t)
    return tamps

def ucc_amplitudes_to_symm_vector(cc, tamps, ipea=0, out=None):
    return ucc_amplitudes_to_vector(cc, pyhast.wick.usymm.ut_symm_amps(tamps, cc.symm_schemes, ipea=ipea), out=out)

def ucc_symm_vector_to_amplitudes(cc, vector, order, has_ref=False, ipea=0):
    return pyhast.wick.usymm.ut_unsymm_amps(ucc_vector_to_amplitudes(cc, vector, order,
        has_ref=has_ref, ipea=ipea), cc.symm_schemes, has_ref=has_ref, ipea=ipea)

def ucc_init_amps(cc):
    import numpy as np
    from pyscf.lib import logger
    from pyscf import lib

    nocca, noccb = cc.n_occ
    fova = cc.ints[0]['ie']
    fovb = cc.ints[1]['IE']
    nvira, nvirb = fova.shape[1], fovb.shape[1]
    mo_ea_o, mo_ea_v = cc.mo_energy[0][:nocca], cc.mo_energy[0][nocca:]
    mo_eb_o, mo_eb_v = cc.mo_energy[1][:noccb], cc.mo_energy[1][noccb:]
    eia_a = lib.direct_sum('i-a->ia', mo_ea_o, mo_ea_v)
    eia_b = lib.direct_sum('i-a->ia', mo_eb_o, mo_eb_v)
    t1a = fova.conj() / eia_a
    t1b = fovb.conj() / eia_b
    eris_oovv = np.array(cc.ints[2]['iiee'])
    eris_oOvV = np.array(cc.ints[3]['iIeE'])
    eris_OOVV = np.array(cc.ints[4]['IIEE'])
    eris = [eris_oovv, eris_oOvV, eris_OOVV]
    ovs = {'i': nocca, 'I': noccb, 'e': nvira, 'E': nvirb}
    for ia, (eri, idx) in enumerate(zip(eris[::2], ['iiee', 'IIEE'])):
        unpk = np.zeros(tuple(ovs[x] for x in idx), dtype=eri.dtype)
        masks, addrs = cc.symm_schemes[1:]
        (omask, vmask), (oaddr, vaddr) = masks[2][ia * 2:ia * 2 + 2], addrs[2][ia * 2:ia * 2 + 2]
        unpk[omask[0, :, None], omask[1, :, None], vmask[0, None], vmask[1, None]] = \
            eri[oaddr[:, None], vaddr[None]]
        unpk = unpk - unpk.transpose((1, 0, 2, 3))
        eris[ia * 2] = unpk - unpk.transpose((0, 1, 3, 2))
    eris_oovv, eris_oOvV, eris_OOVV = eris
    t2aa = np.ascontiguousarray(eris_oovv / lib.direct_sum('ia+jb->ijab', eia_a, eia_a))
    t2ab = np.ascontiguousarray(eris_oOvV / lib.direct_sum('ia+jb->ijab', eia_a, eia_b))
    t2bb = np.ascontiguousarray(eris_OOVV / lib.direct_sum('ia+jb->ijab', eia_b, eia_b))
    e = np.einsum('iJaB,iJaB->', t2ab, eris_oOvV, optimize='optimal')
    e += 0.25 * np.einsum('ijab,ijab->', t2aa, eris_oovv, optimize='optimal')
    e += 0.25 * np.einsum('ijab,ijab->', t2bb, eris_OOVV, optimize='optimal')
    cc.emp2 = e.real
    logger.info(cc, 'Init t2, MP2 energy = %.15g', cc.emp2)
    tamps = [[t1a, t1b], [t2aa, t2ab, t2bb]]
    tamps = pyhast.wick.usymm.ut_symm_amps(tamps, cc.symm_schemes)
    for it in range(3, cc.order + 1):
        t = []
        for k in range(it + 1):
            x = tuple(cc.symm_schemes[0][p][q] for qs in [[0, 2], [1, 3]] for p, q in zip([it - k, k], qs) if p)
            t.append(np.zeros(x, dtype=t1a.dtype))
        tamps.append(t)
    return tamps[:cc.order]

def ucc_zero_amps(cc, order):
    import numpy as np
    (nocca, noccb), (nvira, nvirb), dtype = cc.n_occ, cc.n_virt, cc.ints[0]['ie'].dtype
    tamps = []
    for it in range(1, order + 1):
        t = []
        for k in range(it + 1):
            x = tuple(cc.symm_schemes[0][p][q] for qs in [[0, 2], [1, 3]] for p, q in zip([it - k, k], qs) if p)
            t.append(np.zeros(x, dtype=dtype))
        tamps.append(t)
    return tamps

class UCC:
    def __init__(self, mf, t_order=3, frozen=0, verbose=None, diis=True, diis_symm_amps=False, dtype=float,
                 mo_coeff=None, gen_eq=True, manifest=True, split=True, lift=True, sum_lift=True, use_hint=True,
                 eval_t=None, symm_eval_t=None, dsum_eval_t=None, ddiv_eval_t=None,
                 label_symm=True, newton_krylov=False, level_shift=0.0, gen_lamb_eq=False,
                 gen_npdm_eq=False, lamb_order=None, npdm_order=None,
                 e_order=None, max_comm=None, diis_scratch=None, diis_scratch_start=0,
                 diis_order=None, amps_save_dir=None, symm_amps=True):
        import sys, numpy as np, os
        assert symm_amps == True
        self.order = t_order
        self.manifest, self.split, self.lift = manifest, split, lift
        self.sum_lift, self.use_hint, self.label_symm = sum_lift, use_hint, label_symm
        self.lamb_order = t_order if lamb_order is None else lamb_order
        self.npdm_order = t_order if npdm_order is None else npdm_order
        self.max_comm = max_comm
        self.nfrozen = frozen
        self.e_order = self.order if e_order is None else e_order
        self.e_hf, self.ints, self.n_occ, self.n_virt = ut_ao2mo(mf, nfrozen=frozen, mo=mo_coeff, dtype=dtype)
        self.eqs = self.generate_eqs('amps') if gen_eq else None
        self.lamb_eqs = self.generate_eqs('lamb') if gen_lamb_eq else None
        self.npdm_eqs = self.generate_eqs('npdm') if gen_npdm_eq else None
        self.eval_t = eval_t
        self.symm_eval_t = symm_eval_t
        self.dsum_eval_t = dsum_eval_t
        self.ddiv_eval_t = ddiv_eval_t
        self.diis_symm_amps, self.diis_scratch, self.diis_scratch_start = diis_symm_amps, diis_scratch, diis_scratch_start
        self.diis_order, self.amps_save_dir = diis_order, amps_save_dir
        if self.amps_save_dir is not None:
            os.makedirs(self.amps_save_dir, exist_ok=True)
        self.symm_t = True
        self.newton_krylov = newton_krylov
        self.symm_schemes, cistrs = pyhast.wick.usymm.ut_symm_schemes(self.n_occ, self.n_virt,
            max(self.e_order, self.order, self.ints[-1].ndim // 2))
        self.merge_schemes = pyhast.wick.usymm.ut_merge_schemes(self.n_occ, self.n_virt, self.order, cistrs) if self.eval_t is None else None
        nocca, noccb = self.n_occ
        faii, faee = np.diag(self.ints[0])[:nocca], np.diag(self.ints[0])[nocca:] + level_shift
        fbii, fbee = np.diag(self.ints[1])[:noccb], np.diag(self.ints[1])[noccb:] + level_shift
        masks, addrs = self.symm_schemes[1:]
        self.exxs = [[] for _ in range(self.order + 1)]
        for kk in range(self.order + 1):
            for i, fxx in enumerate([faii, faee, fbii, fbee]):
                self.exxs[kk].append(np.zeros(masks[kk][i].shape[1], dtype=fxx.dtype))
                self.exxs[kk][i][addrs[kk][i]] = np.sum(fxx[masks[kk][i]], axis=0)
        self.mo_energy = np.array([np.diag(self.ints[0]), np.diag(self.ints[1])])
        self.ints = pyhast.wick.usymm.ut_symm_integral(self.ints, self.n_occ, True, self.symm_schemes)
        self.level_shift = level_shift
        self.stdout = sys.stdout
        self.verbose = mf.verbose if verbose is None else verbose
        self.diis = diis
        self.diis_space = 6
        self.diis_start_cycle = 0
        self.iterative_damping = 1.0
        self.name = "UCC%s" % ("SDTQPH789"[:self.order] if self.order < 10 else self.order)
        self.converged = False
        if self.eqs is not None and self.eval_t in ['hastctr-dfs', 'hastctr-work-dfs'] and self.verbose >= 4:
            pyhast.wick.uwick.ut_symm_dfs_sum_lift_summary(self.eqs, self.n_occ, self.n_virt, (0, 0))
        if self.lamb_eqs is not None and self.eval_t in ['hastctr-dfs', 'hastctr-work-dfs'] and self.verbose >= 4:
            pyhast.wick.uwick.ut_symm_dfs_sum_lift_summary(self.lamb_eqs, self.n_occ, self.n_virt, (0, 0))
    def generate_eqs(self, eq_name, ipea=0, left=False):
        prim_eqs = cc_prim_eqs(self.order, max_comm=self.max_comm, norm=False,
            l_ord=None if eq_name not in ['lamb', 'npdm'] else self.lamb_order,
            p_ord=None if eq_name not in ['npdm'] else self.npdm_order)
        x_ord = {'lamb': self.lamb_order, 'npdm': self.npdm_order}.get(eq_name, self.order)
        real_names = 'HLTR' if eq_name == 'npdm' else 'HTX'
        eqs = pyhast.wick.uwick.ut_simplify_eqs(self.order, prim_eqs, x_ord=x_ord, adapt_ipea=ipea,
            ipea_names='RX',
            manifest=(self.manifest and not self.split), real_names=real_names)
        return pyhast.wick.glift.gt_optimize_eqs(eqs, meq_f=pyhast.wick.uwick.ut_deq_to_meq,
            mani_f=pyhast.wick.uwick.ut_meq_manifest, symm_f=pyhast.wick.gsymm.gt_label_symm,
            manifest=self.manifest, split=self.split, lift=self.lift, sum_lift=self.sum_lift,
            use_hint=self.use_hint, label_symm=self.label_symm, split_lift=(eq_name in ['amps', 'lamb']),
            shapes={'I': self.n_occ[0] + self.n_occ[1], 'E': self.n_virt[0] + self.n_virt[1]})
    energy = ucc_energy
    zero_amps = ucc_zero_amps
    update_amps = ucc_update_amps
    update_lambda = lambda cc, lamps, tamps: ucc_update_amps(cc, tamps, lamps=lamps)

    npdm_intermediates = ucc_npdm_intermediates
    init_amps = ucc_init_amps
    amplitudes_to_vector = ucc_amplitudes_to_vector
    vector_to_amplitudes = ucc_vector_to_amplitudes
    amplitudes_to_symm_vector = ucc_amplitudes_to_symm_vector
    symm_vector_to_amplitudes = ucc_symm_vector_to_amplitudes
    get_init_guess = ucc_init_amps
    run_diis = cc_run_diis
    kernel = cc_kernel
    solve_lambda = cc_lamb_kernel
    make_npdms = cc_npdm_kernel

    e_tot = property(lambda self: self.e_hf + self.e_corr)

def UCCSD(*args, **kwargs):
    kwargs["t_order"] = 2
    return UCC(*args, **kwargs)

def UCCSDT(*args, **kwargs):
    kwargs["t_order"] = 3
    return UCC(*args, **kwargs)

def UCCSDTQ(*args, **kwargs):
    kwargs["t_order"] = 4
    return UCC(*args, **kwargs)
