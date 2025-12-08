
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

class DIIS:
    def __init__(self, dev=None, space=6, scratch=None, scratch_start=0):
        self.verbose, self.stdout = dev.verbose, dev.stdout
        self.dev = dev
        self.space = space
        self.xs, self.es = [None] * space, [None] * space
        self._head, self._H, self._xprev = 0, None, None
        self.scratch, self.scratch_start = scratch, scratch_start

    def update(self, x):
        import numpy as np
        if self._xprev is None:
            self._xprev = x
            return x
        self.xs[self._head], self.es[self._head] = x, x - self._xprev
        if self.scratch is not None and self._head >= self.scratch_start:
            np.save('%s/DIIS-%s-X%02d.npy' % (self.scratch, self.dev.name, self._head), self.xs[self._head])
            np.save('%s/DIIS-%s-E%02d.npy' % (self.scratch, self.dev.name, self._head), self.es[self._head])
        nd = len([v for v in self.xs if v is not None])

        if self._H is None:
            self._H = np.zeros((self.space + 1, self.space + 1), x.dtype)
            self._H[0, 1:] = self._H[1:, 0] = 1
        for i in range(nd):
            if self.scratch is not None and i >= self.scratch_start:
                ei = np.load('%s/DIIS-%s-E%02d.npy' % (self.scratch, self.dev.name, i))
            else:
                ei = self.es[i]
            self._H[self._head + 1, i + 1] = np.dot(self.es[self._head], ei)
            self._H[i + 1, self._head + 1] = self._H[self._head + 1, i + 1].conj()
            ei = None
        if self.scratch is not None and self._head >= self.scratch_start:
            self.xs[self._head], self.es[self._head] = (), ()
        self._head = (self._head + 1) % self.space
        self._xprev = None
        self._xprev = self.extrapolate(nd)
        return self._xprev.reshape(x.shape)

    def clean_scratch(self):
        import os
        if self.scratch is not None:
            for i, x in enumerate(self.xs):
                if i >= self.scratch_start and x == ():
                    os.remove('%s/DIIS-%s-X%02d.npy' % (self.scratch, self.dev.name, i))
            for i, x in enumerate(self.es):
                if i >= self.scratch_start and x == ():
                    os.remove('%s/DIIS-%s-E%02d.npy' % (self.scratch, self.dev.name, i))

    def extrapolate(self, nd):
        import numpy as np
        from pyscf.lib import logger

        h = self._H[:nd + 1, :nd + 1]
        g = np.zeros(nd + 1, h.dtype)
        g[0] = 1

        w, v = np.linalg.eigh(h)
        if np.any(abs(w) < 1E-14):
            logger.debug(self, 'Linear dependence found in DIIS error vectors.')
            idx = abs(w) > 1E-14
            c = np.dot(v[:, idx] * (1 / w[idx]), np.dot(v[:, idx].T.conj(), g))
        else:
            c = np.linalg.solve(h, g)
        logger.debug1(self, 'diis-c %s', c)

        xnew = None
        for i, ci in enumerate(c[1:]):
            if self.scratch is not None and i >= self.scratch_start:
                xi = np.load('%s/DIIS-%s-X%02d.npy' % (self.scratch, self.dev.name, i))
            else:
                xi = self.xs[i]
            if xnew is None:
                xnew = np.zeros((xi.size, ), xi.dtype)
            xnew += xi * ci
            xi = None
        return xnew
