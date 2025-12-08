
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

def ut_reorder(ints, reord):
    import numpy as np
    reorda, reordb = reord
    h1e_a = np.array(ints[0][reorda][:, reorda], copy=True)
    h1e_b = np.array(ints[1][reordb][:, reordb], copy=True)
    xg2e_aa = np.array(ints[2][reorda][:, reorda][:, :, reorda][:, :, :, reorda], copy=True)
    xg2e_ab = np.array(ints[3][reorda][:, reordb][:, :, reorda][:, :, :, reordb], copy=True)
    xg2e_bb = np.array(ints[4][reordb][:, reordb][:, :, reordb][:, :, :, reordb], copy=True)
    return h1e_a, h1e_b, xg2e_aa, xg2e_ab, xg2e_bb

def ut_normal_order(n_occ, ints, ecore, anti_g2e=True, phys_g2e=True):
    import numpy as np
    nocca, noccb = n_occ
    h1e_a, h1e_b, g2e_aa, g2e_ab, g2e_bb = ints
    if phys_g2e:
        g2e_aa = g2e_aa.transpose(0, 2, 1, 3)
        g2e_ab = g2e_ab.transpose(0, 2, 1, 3)
        g2e_bb = g2e_bb.transpose(0, 2, 1, 3)
    ecore += np.einsum("jj->", h1e_a[:nocca, :nocca], optimize='optimal')
    ecore += np.einsum("jj->", h1e_b[:noccb, :noccb], optimize='optimal')
    ecore += 0.5 * np.einsum('iijj->', g2e_aa[:nocca, :nocca, :nocca, :nocca], optimize='optimal')
    ecore += 0.5 * np.einsum('iijj->', g2e_bb[:noccb, :noccb, :noccb, :noccb], optimize='optimal')
    ecore += 1.0 * np.einsum('iijj->', g2e_ab[:nocca, :nocca, :noccb, :noccb], optimize='optimal')
    if not anti_g2e:
        ecore -= 0.5 * np.einsum('ijji->', g2e_aa[:nocca, :nocca, :nocca, :nocca], optimize='optimal')
        ecore -= 0.5 * np.einsum('ijji->', g2e_bb[:noccb, :noccb, :noccb, :noccb], optimize='optimal')
    h1e_a, h1e_b = h1e_a.copy(), h1e_b.copy()
    h1e_a += np.einsum("mnjj->mn", g2e_aa[:, :, :nocca, :nocca], optimize='optimal')
    h1e_a += np.einsum("mnjj->mn", g2e_ab[:, :, :noccb, :noccb], optimize='optimal')
    h1e_b += np.einsum("mnjj->mn", g2e_bb[:, :, :noccb, :noccb], optimize='optimal')
    h1e_b += np.einsum("jjmn->mn", g2e_ab[:nocca, :nocca, :, :], optimize='optimal')
    if not anti_g2e:
        h1e_a -= np.einsum("mjjn->mn", g2e_aa[:, :nocca, :nocca, :], optimize='optimal')
        h1e_b -= np.einsum("mjjn->mn", g2e_bb[:, :noccb, :noccb, :], optimize='optimal')
    if phys_g2e:
        g2e_aa = g2e_aa.transpose(0, 2, 1, 3)
        g2e_ab = g2e_ab.transpose(0, 2, 1, 3)
        g2e_bb = g2e_bb.transpose(0, 2, 1, 3)
    return (h1e_a, h1e_b, g2e_aa, g2e_ab, g2e_bb), ecore

def ut_ao2mo(mf, nfrozen=0, normal_ord=True, anti_g2e=True, ncas=None, n_occ=None, mo=None, dtype=float):
    from pyscf import ao2mo
    import numpy as np

    mo = [mo, mo] if mo is not None and np.array(mo).ndim == 2 else mo
    mol = mf.mol
    ncore = nfrozen
    mo_a, mo_b = mf.mo_coeff if mo is None else mo
    ncas = mo_a.shape[1] - ncore if ncas is None else ncas

    nocca = (mol.nelectron + mol.spin) // 2 - nfrozen if n_occ is None else n_occ[0] - nfrozen
    noccb = (mol.nelectron - mol.spin) // 2 - nfrozen if n_occ is None else n_occ[1] - nfrozen
    nvira, nvirb = ncas - nocca, ncas - noccb

    mo_core = mo_a[:, :ncore], mo_b[:, :ncore]
    mo_cas = mo_a[:, ncore : ncore + ncas], mo_b[:, ncore : ncore + ncas]
    hcore_ao = mf.get_hcore()
    hveff_ao = (0, 0)

    if ncore != 0:
        core_dmao = mo_core[0] @ mo_core[0].T.conj(), mo_core[1] @ mo_core[1].T.conj()
        vj, vk = mf.get_jk(mol, core_dmao)
        hveff_ao = vj[0] + vj[1] - vk
        ecore0 = np.einsum("ij,ji->", core_dmao[0], hcore_ao + 0.5 * hveff_ao[0], optimize='optimal')
        ecore0 += np.einsum("ij,ji->", core_dmao[1], hcore_ao + 0.5 * hveff_ao[1], optimize='optimal')
    else:
        ecore0 = 0.0

    if hcore_ao.ndim == 3:
        h1e_a = mo_cas[0].T.conj() @ (hcore_ao[0] + hveff_ao[0]) @ mo_cas[0]
        h1e_b = mo_cas[1].T.conj() @ (hcore_ao[1] + hveff_ao[1]) @ mo_cas[1]
    else:
        h1e_a = mo_cas[0].T.conj() @ (hcore_ao + hveff_ao[0]) @ mo_cas[0]
        h1e_b = mo_cas[1].T.conj() @ (hcore_ao + hveff_ao[1]) @ mo_cas[1]
    mo_a, mo_b = mo_cas

    eri_ao = mol if mf._eri is None else mf._eri
    if eri_ao.ndim == 5:
        g2e_aa = ao2mo.restore(1, ao2mo.full(eri_ao[0], mo_a), ncas)
        g2e_ab = ao2mo.restore(1, ao2mo.general(eri_ao[1], (mo_a, mo_a, mo_b, mo_b)), ncas)
        g2e_bb = ao2mo.restore(1, ao2mo.full(eri_ao[2], mo_b), ncas)
    else:
        g2e_aa = ao2mo.restore(1, ao2mo.full(eri_ao, mo_a), ncas)
        g2e_ab = ao2mo.restore(1, ao2mo.general(eri_ao, (mo_a, mo_a, mo_b, mo_b)), ncas)
        g2e_bb = ao2mo.restore(1, ao2mo.full(eri_ao, mo_b), ncas)

    h1e_a = np.asarray(h1e_a, dtype=dtype)
    h1e_b = np.asarray(h1e_b, dtype=dtype)
    g2e_aa = np.asarray(g2e_aa, dtype=dtype)
    g2e_ab = np.asarray(g2e_ab, dtype=dtype)
    g2e_bb = np.asarray(g2e_bb, dtype=dtype)

    enuc = mol.energy_nuc() + ecore0
    ecore = enuc

    if normal_ord:
        (h1e_a, h1e_b, g2e_aa, g2e_ab, g2e_bb), ecore = ut_normal_order((nocca, noccb),
            (h1e_a, h1e_b, g2e_aa, g2e_ab, g2e_bb), ecore, anti_g2e=False, phys_g2e=False)

    if g2e_aa.ndim == 4 and anti_g2e:
        xg2e_aa = g2e_aa.transpose(0, 2, 1, 3) - g2e_aa.transpose(0, 2, 3, 1)
        xg2e_bb = g2e_bb.transpose(0, 2, 1, 3) - g2e_bb.transpose(0, 2, 3, 1)
    else:
        xg2e_aa = g2e_aa.transpose(0, 2, 1, 3) if g2e_aa.ndim == 4 else g2e_aa
        xg2e_bb = g2e_bb.transpose(0, 2, 1, 3) if g2e_bb.ndim == 4 else g2e_bb

    xg2e_ab = g2e_ab.transpose(0, 2, 1, 3) if g2e_aa.ndim == 4 else g2e_ab
    return ecore, (h1e_a, h1e_b, xg2e_aa, xg2e_ab, xg2e_bb), (nocca, noccb), (nvira, nvirb)

def ut_energy_from_pdms(ecore, ints, pdms):
    import numpy as np
    ener = ecore + np.einsum('ij,ij->', ints[0], pdms[0][0], optimize='optimal') \
        + np.einsum('ij,ij->', ints[1], pdms[0][1], optimize='optimal')
    ener += 0.25 * np.einsum('ijkl,ijkl->', ints[2], pdms[1][0], optimize='optimal') \
        + np.einsum('ijkl,ijkl->', ints[3], pdms[1][1], optimize='optimal') \
        + 0.25 * np.einsum('ijkl,ijkl->', ints[4], pdms[1][2], optimize='optimal')
    return ener
