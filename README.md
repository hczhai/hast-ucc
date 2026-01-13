
HAST-UCC
========

A hasty implementation for arbitrary-order spin-unrestricted coupled cluster, supporting up to 128 occupied and 128 virtual orbitals.

To compile the C++ implementation for symmetrized tensor contraction (need at least g++ version 10), use:

```bash
mkdir build
cd build
cmake ..
make
```

An example script for UCCSDTQP:

```python
from pyscf import gto, scf
import pyhast

mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto3g", symmetry="d2h", verbose=0)

mf = scf.UHF(mol)
mf.kernel()

cc_opts = dict(eval_t='hastctr')
mcc = pyhast.sr.ucc.UCC(mf, t_order=5, verbose=4, diis=True, **cc_opts)
mcc.kernel() # E[UCCSDTQP] = -107.65410283385334
```

Reference:

Zhai, Huanchen, et al. "Classical solution of the FeMo-cofactor model to chemical accuracy and its implications." arXiv preprint [arXiv:2601.04621](https://arxiv.org/abs/2601.04621) (2026).
