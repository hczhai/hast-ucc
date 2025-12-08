
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

try:

    import ctypes, os
    _path = os.path.dirname(__file__)
    _lib = ctypes.CDLL(os.path.join(_path, "..", "..", "build", "libhast-ctr.so"))

    import numpy as np

    _lib.tensordot.argtypes = (
        ctypes.c_int16, # ndim_a
        ctypes.c_int16, # ndim_b
        ctypes.POINTER(ctypes.c_size_t), # shape_a
        ctypes.POINTER(ctypes.c_size_t), # shape_b
        ctypes.POINTER(ctypes.c_size_t), # strides_a
        ctypes.POINTER(ctypes.c_size_t), # strides_b
        ctypes.c_int16, # nctr
        ctypes.POINTER(ctypes.c_int16), # idx_a
        ctypes.POINTER(ctypes.c_int16), # idx_b
        ctypes.POINTER(ctypes.c_int16), # tr_c
        ctypes.POINTER(ctypes.c_double), # a
        ctypes.POINTER(ctypes.c_double), # b
        ctypes.POINTER(ctypes.c_double), # c
        ctypes.c_double, # alpha
        ctypes.c_double, # beta
        ctypes.POINTER(ctypes.c_size_t), # strides_c
        ctypes.POINTER(ctypes.c_int16), # symm_desc_a
        ctypes.POINTER(ctypes.c_int16), # symm_desc_b
        ctypes.POINTER(ctypes.c_int16), # symm_desc_c
    )
    _lib.transpose.argtypes = (
        ctypes.c_int16, # ndim_a
        ctypes.POINTER(ctypes.c_size_t), # shape_a
        ctypes.POINTER(ctypes.c_size_t), # strides_a
        ctypes.POINTER(ctypes.c_size_t), # strides_b
        ctypes.POINTER(ctypes.c_int16), # tr_b
        ctypes.POINTER(ctypes.c_double), # a
        ctypes.POINTER(ctypes.c_double), # b
        ctypes.c_double, # alpha
        ctypes.c_double, # beta
        ctypes.POINTER(ctypes.c_int16), # symm_desc_a
        ctypes.POINTER(ctypes.c_int16), # symm_desc_b
    )
    _lib.tensordot_nflops.argtypes = (
        ctypes.c_int16, # ndim_a
        ctypes.c_int16, # ndim_b
        ctypes.POINTER(ctypes.c_size_t), # shape_a
        ctypes.POINTER(ctypes.c_size_t), # shape_b
        ctypes.POINTER(ctypes.c_size_t), # strides_a
        ctypes.POINTER(ctypes.c_size_t), # strides_b
        ctypes.c_int16, # nctr
        ctypes.POINTER(ctypes.c_int16), # idx_a
        ctypes.POINTER(ctypes.c_int16), # idx_b
        ctypes.POINTER(ctypes.c_int16), # tr_c
        ctypes.POINTER(ctypes.c_size_t), # strides_c
        ctypes.POINTER(ctypes.c_int16), # symm_desc_a
        ctypes.POINTER(ctypes.c_int16), # symm_desc_b
        ctypes.POINTER(ctypes.c_int16), # symm_desc_c
    )
    _lib.reset_timer.argtypes = ()
    _lib.check_timer.argtypes = (
        ctypes.POINTER(ctypes.c_double), # tsymm
        ctypes.POINTER(ctypes.c_double), # tgemm
    )
    _lib.tensordot.restype = ctypes.c_int
    _lib.transpose.restype = ctypes.c_int
    _lib.reset_timer.restype = ctypes.c_int
    _lib.check_timer.restype = ctypes.c_int

    def _tensordot(ndim_a, ndim_b, shape_a, shape_b, strides_a, strides_b, nctr, idx_a, idx_b, tr_c, a, b, c, alpha,
                   beta, strides_c, symm_a, symm_b, symm_c):
        return _lib.tensordot(
            ctypes.c_int16(ndim_a),
            ctypes.c_int16(ndim_b),
            np.array(shape_a, dtype=int).ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            np.array(shape_b, dtype=int).ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            np.array(strides_a, dtype=int).ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            np.array(strides_b, dtype=int).ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            ctypes.c_int16(nctr),
            np.array(idx_a, dtype=np.int16).ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            np.array(idx_b, dtype=np.int16).ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            np.array(tr_c, dtype=np.int16).ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_double(alpha),
            ctypes.c_double(beta),
            np.array(strides_c, dtype=int).ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            np.array(symm_a, dtype=np.int16).ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            np.array(symm_b, dtype=np.int16).ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            np.array(symm_c, dtype=np.int16).ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        )

    def _transpose(ndim_a, shape_a, strides_a, strides_b, tr_b, a, b, alpha,
                   beta,  symm_a, symm_b):
        return _lib.transpose(
            ctypes.c_int16(ndim_a),
            np.array(shape_a, dtype=int).ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            np.array(strides_a, dtype=int).ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            np.array(strides_b, dtype=int).ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            np.array(tr_b, dtype=np.int16).ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_double(alpha),
            ctypes.c_double(beta),
            np.array(symm_a, dtype=np.int16).ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            np.array(symm_b, dtype=np.int16).ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        )
    
    def _tensordot_nflops(ndim_a, ndim_b, shape_a, shape_b, strides_a, strides_b, nctr, idx_a, idx_b, tr_c,
                   strides_c, symm_a, symm_b, symm_c):
        return _lib.tensordot_nflops(
            ctypes.c_int16(ndim_a),
            ctypes.c_int16(ndim_b),
            np.array(shape_a, dtype=int).ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            np.array(shape_b, dtype=int).ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            np.array(strides_a, dtype=int).ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            np.array(strides_b, dtype=int).ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            ctypes.c_int16(nctr),
            np.array(idx_a, dtype=np.int16).ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            np.array(idx_b, dtype=np.int16).ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            np.array(tr_c, dtype=np.int16).ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            np.array(strides_c, dtype=int).ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            np.array(symm_a, dtype=np.int16).ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            np.array(symm_b, dtype=np.int16).ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            np.array(symm_c, dtype=np.int16).ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        )
    
    def reset_timer():
        return _lib.reset_timer()
    
    def check_timer():
        tsymm, tgemm = np.zeros((1, )), np.zeros((1, ))
        _lib.check_timer(
            tsymm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            tgemm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return float(tsymm[0]), float(tgemm[0])

except OSError:
    pass

def symm_tensor_shape(n_occ, n_virt, n_cas, ndim, xsymm):
    xshape = [0] * ndim
    for x in xsymm:
        for y in x[2:]:
            if not isinstance(n_cas, int):
                xshape[y] = [n_occ[0], n_virt[0], n_occ[1], n_virt[1]][x[1]]
            else:
                xshape[y] = [n_occ, n_virt][x[1]]
    xsymm_shape = []
    for p in xsymm:
        x = 1
        for it, q in enumerate(p[2:]):
            x = x * (xshape[q] - it) // (it + 1)
        xsymm_shape.append(x)
    return tuple(xsymm_shape)

def symm_contract_nflops(n_occ, n_virt, n_cas, script, xsymms):
    scrs = script.split('->')[0].split(',') + script.split('->')[1:]
    shapes = [[0] * len(x) for x in scrs]
    for ix, xsm in enumerate(xsymms):
        for x in xsm:
            for y in x[2:]:
                if not isinstance(n_cas, int):
                    shapes[ix][y] = [n_occ[0], n_virt[0], n_occ[1], n_virt[1]][x[1]]
                else:
                    shapes[ix][y] = [n_occ, n_virt][x[1]]
    symm_shapes = [[] for _ in scrs]
    for ix, (xsm, xshape) in enumerate(zip(xsymms, shapes)):
        for p in xsm:
            x = 1
            for it, q in enumerate(p[2:]):
                x = x * (xshape[q] - it) // (it + 1)
            symm_shapes[ix].append(x)
    symm_strides = [[1] * len(x)  for x in symm_shapes]
    strides = []
    for xscr, xsymm, xsstr in zip(scrs, xsymms, symm_strides):
        strides.append([xsstr[[j for j, jx in enumerate(xsymm) if i in jx[2:]][0]] for i in range(len(xscr))])
    xsymmvs = [[x for xx in xsymm for x in (-xx[0], ) + xx[2:]] + [0] for xsymm in xsymms]
    if len(scrs) == 3:
        idx_a, idx_b = [], []
        for ia, ax in enumerate(scrs[0]):
            for ib, bx in enumerate(scrs[1]):
                if ax == bx:
                    idx_a.append(ia)
                    idx_b.append(ib)
        scr_c = ''
        for ia, ax in enumerate(scrs[0]):
            for ib, bx in enumerate(scrs[2]):
                if ax == bx:
                    scr_c += ax
        for ia, ax in enumerate(scrs[1]):
            for ib, bx in enumerate(scrs[2]):
                if ax == bx:
                    scr_c += ax
        tr_c = [scr_c.index(cx) for cx in scrs[2]]
        return _tensordot_nflops(len(shapes[0]), len(shapes[1]), shapes[0], shapes[1], strides[0], strides[1], len(idx_a), idx_a, idx_b,
            tr_c, strides[2], xsymmvs[0], xsymmvs[1], xsymmvs[2])
    else:
        return 0

def symm_contract(n_occ, n_virt, n_cas, script, xsymms, *tensors, alpha=1.0, beta=0.0, out=None):
    scrs = script.split('->')[0].split(',') + script.split('->')[1:]
    shapes = [[0] * len(x) for x in scrs]
    for ix, xsm in enumerate(xsymms):
        for x in xsm:
            for y in x[2:]:
                if not isinstance(n_cas, int):
                    shapes[ix][y] = [n_occ[0], n_virt[0], n_occ[1], n_virt[1]][x[1]]
                else:
                    shapes[ix][y] = [n_occ, n_virt][x[1]]
    symm_shapes = [[] for _ in scrs]
    for ix, (xsm, xshape) in enumerate(zip(xsymms, shapes)):
        for p in xsm:
            x = 1
            for it, q in enumerate(p[2:]):
                x = x * (xshape[q] - it) // (it + 1)
            symm_shapes[ix].append(x)
    symm_strides = [[1] for _ in scrs]
    if out is None:
        out = np.empty(symm_shapes[-1])
        assert beta == 0.0
    elif isinstance(out, np.float64) or isinstance(out, np.complex128):
        out = np.array(out)
    for ix, ts in enumerate(list(tensors) + [out]):
        assert ts.shape == tuple(symm_shapes[ix])
        symm_strides[ix] = list(x / 8 for x in ts.strides)
    strides = []
    for xscr, xsymm, xsstr in zip(scrs, xsymms, symm_strides):
        strides.append([xsstr[[j for j, jx in enumerate(xsymm) if i in jx[2:]][0]] for i in range(len(xscr))])
    xsymmvs = [[x for xx in xsymm for x in (-xx[0], ) + xx[2:]] + [0] for xsymm in xsymms]
    if len(tensors) == 2:
        idx_a, idx_b = [], []
        for ia, ax in enumerate(scrs[0]):
            for ib, bx in enumerate(scrs[1]):
                if ax == bx:
                    idx_a.append(ia)
                    idx_b.append(ib)
        scr_c = ''
        for ia, ax in enumerate(scrs[0]):
            for ib, bx in enumerate(scrs[2]):
                if ax == bx:
                    scr_c += ax
        for ia, ax in enumerate(scrs[1]):
            for ib, bx in enumerate(scrs[2]):
                if ax == bx:
                    scr_c += ax
        tr_c = [scr_c.index(cx) for cx in scrs[2]]
        _tensordot(len(shapes[0]), len(shapes[1]), shapes[0], shapes[1], strides[0], strides[1], len(idx_a), idx_a, idx_b,
            tr_c, tensors[0], tensors[1], out, alpha, beta, strides[2], xsymmvs[0], xsymmvs[1], xsymmvs[2])
    else:
        tr_b = [scrs[0].index(cx) for cx in scrs[1]]
        _transpose(len(shapes[0]), shapes[0], strides[0], strides[1], tr_b, tensors[0], out, alpha, beta, xsymmvs[0], xsymmvs[1])
    return out
