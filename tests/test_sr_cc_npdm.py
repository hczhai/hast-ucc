
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

import pytest
import pyhast

@pytest.fixture(scope="module", params=[2, 3])
def t_ord(request):
    return request.param

@pytest.fixture(scope="module", params=['UHF'])
def spin_symm(request):
    return request.param

@pytest.fixture(scope="module", params=[[True, True, True, 'hastctr']])
def symbolic_opts(request):
    return request.param

@pytest.fixture(scope="module", params=[0, 1])
def nfrozen(request):
    return request.param

class TestSRCCNPDM:

    def test_cc_npdm(self, spin_symm, t_ord, symbolic_opts, nfrozen):

        from pyscf import gto, scf, cc
        import numpy as np
        split, lift, _, eval_t = symbolic_opts

        mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='sto3g', verbose=3)
        mf = scf.UHF(mol).run(conv_tol=1E-14)
        mcc = pyhast.sr.ucc.UCC(mf, t_order=t_ord, verbose=4, diis=True, split=split, diis_symm_amps=False, label_symm=True,
            lift=lift, eval_t=eval_t, gen_lamb_eq=True, gen_npdm_eq=True, npdm_order=2, frozen=nfrozen)
        xcc = cc.UCCSD(mf, frozen=nfrozen)
        mcc.kernel(tol=1E-14, max_cycle=200)
        assert mcc.converged
        assert abs(mf.e_tot - -74.9611711378677) < 1E-10

        std_energies = [[-75.01913630605031, -75.01922418629151], [-75.0190636841548, -75.0191517896563]]
        assert abs(mcc.e_tot - std_energies[nfrozen][t_ord - 2]) < 1E-8

        lamb_converged, lamps = mcc.solve_lambda(tol=1E-12)
        assert lamb_converged
        dm1, dm2 = mcc.make_npdms(order=2)

        lamps = pyhast.wick.usymm.ut_unsymm_amps(lamps, mcc.symm_schemes)

        if t_ord == 2:
            xcc.conv_tol_normt = 1E-12
            xcc.max_cycle = 200
            xcc.kernel()
            xl1, xl2 = xcc.solve_lambda()
            xdm1 = xcc.make_rdm1()
            xdm2 = xcc.make_rdm2()
            assert np.sum([np.linalg.norm(x - y) for x, y in zip(xl1, lamps[0])]) < 1E-6
            assert np.sum([np.linalg.norm(x - y) for x, y in zip(xl2, lamps[1])]) < 1E-5

            assert abs(mcc.e_tot - xcc.e_tot) < 1E-8
            assert np.linalg.norm(np.array(dm1) - np.array(xdm1)) < 1E-6
            assert np.linalg.norm(np.array(dm2) - np.array(xdm2)) < 1E-5

        ecore, ints = pyhast.sr.ao2mo.ut_ao2mo(mf, nfrozen=0, mo=mf.mo_coeff, normal_ord=False)[:2]
        exx = pyhast.sr.ao2mo.ut_energy_from_pdms(ecore, ints, [dm1, [dm.transpose(0, 2, 1, 3) for dm in dm2]])
        assert abs(exx - mcc.e_tot) < 1E-8

        mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='sto3g', verbose=3, spin=2)
        mf = scf.UHF(mol).run(conv_tol=1E-14)
        mcc = pyhast.sr.ucc.UCC(mf, t_order=t_ord, verbose=4, diis=True, split=split, diis_symm_amps=False, label_symm=True,
            lift=lift, eval_t=eval_t, gen_lamb_eq=True, gen_npdm_eq=True, npdm_order=2, frozen=nfrozen)
        xcc = cc.UCCSD(mf, frozen=nfrozen)

        mcc.kernel(tol=1E-14, max_cycle=200)
        assert mcc.converged
        assert abs(mf.e_tot - -74.60916362689153) < 1E-10

        std_energies = [[-74.64806351203163, -74.64844035461576], [-74.64802361293438, -74.64839993160152]]
        assert abs(mcc.e_tot - std_energies[nfrozen][t_ord - 2]) < 1E-8

        lamb_converged, lamps = mcc.solve_lambda(tol=1E-12)
        assert lamb_converged
        dm1, dm2 = mcc.make_npdms(order=2)

        lamps = pyhast.wick.usymm.ut_unsymm_amps(lamps, mcc.symm_schemes)

        if t_ord == 2:
            xcc.conv_tol_normt = 1E-12
            xcc.max_cycle = 200
            xcc.kernel()
            xl1, xl2 = xcc.solve_lambda()
            xdm1 = xcc.make_rdm1()
            xdm2 = xcc.make_rdm2()
            assert np.sum([np.linalg.norm(x - y) for x, y in zip(xl1, lamps[0])]) < 1E-6
            assert np.sum([np.linalg.norm(x - y) for x, y in zip(xl2, lamps[1])]) < 1E-5

            assert abs(mcc.e_tot - xcc.e_tot) < 1E-8
            assert np.linalg.norm(np.array(dm1) - np.array(xdm1)) < 1E-6
            assert np.linalg.norm(np.array(dm2) - np.array(xdm2)) < 1E-5

        ecore, ints = pyhast.sr.ao2mo.ut_ao2mo(mf, nfrozen=0, mo=mf.mo_coeff, normal_ord=False)[:2]
        exx = pyhast.sr.ao2mo.ut_energy_from_pdms(ecore, ints, [dm1, [dm.transpose(0, 2, 1, 3) for dm in dm2]])
        assert abs(exx - mcc.e_tot) < 1E-8
