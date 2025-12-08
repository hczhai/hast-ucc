
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

@pytest.fixture(scope="module", params=[[True, False, True, 'hastctr']])
def symbolic_opts(request):
    return request.param

@pytest.fixture(scope="module", params=[0, 1])
def nfrozen(request):
    return request.param

@pytest.fixture(scope="module", params=[False])
def newton(request):
    return request.param

@pytest.fixture(scope="module", params=[False, True])
def diis(request):
    return request.param

class TestSRCC:

    def test_cc(self, spin_symm, t_ord, symbolic_opts, nfrozen, newton, diis):

        from pyscf import gto, scf
        split, lift, _, eval_t = symbolic_opts

        mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='sto3g', verbose=3)
        mf = scf.UHF(mol).run(conv_tol=1E-14)
        mcc = pyhast.sr.ucc.UCC(mf, t_order=t_ord, verbose=4, diis=diis, frozen=nfrozen, label_symm=True,
            split=split, lift=lift, eval_t=eval_t, newton_krylov=newton)

        mcc.kernel(tol=1E-12)
        assert mcc.converged
        assert abs(mf.e_tot - -74.9611711378677) < 1E-10
        if t_ord == 2:
            assert abs(mcc.e_tot - [-75.0191363060504, -75.0190636841548][nfrozen]) < 1E-8
        else:
            assert abs(mcc.e_tot - [-75.0192241862915, -75.0191517896563][nfrozen]) < 1E-8

        mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='sto3g', verbose=3, spin=2)
        mf = scf.UHF(mol).run(conv_tol=1E-14)
        mcc = pyhast.sr.ucc.UCC(mf, t_order=t_ord, verbose=4, diis=diis, frozen=nfrozen, label_symm=True,
            split=split, lift=lift, newton_krylov=newton)

        mcc.kernel(tol=1E-12, max_cycle=100)
        assert mcc.converged
        assert abs(mf.e_tot - -74.60916362689153) < 1E-10
        if t_ord == 2:
            assert abs(mcc.e_tot - [-74.64806351203163, -74.64802361293438][nfrozen]) < 1E-8
        else:
            assert abs(mcc.e_tot - [-74.64844035461576, -74.64839993160152][nfrozen]) < 1E-8
