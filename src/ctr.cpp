
/*
 *  hast-ucc: a hasty implementation for spin-unrestricted coupled cluster
 *  Copyright (C) 2025 Huanchen Zhai <hczhai.ok@gmail.com>
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 */

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/time.h>
#include <thread>
#include <utility>

struct Timer {
    double current;
    Timer() : current(0) {}
    double get_time() {
        struct timeval t;
        gettimeofday(&t, NULL);
        double previous = current;
        current = t.tv_sec + 1E-6 * t.tv_usec;
        return current - previous;
    }
};

// using ubit_t = uint64_t;
using ubit_t = unsigned __int128;

static inline int popcount128(ubit_t x) {
    if constexpr (sizeof(ubit_t) == 16) {
        const uint64_t *p = (const uint64_t *)&x;
        return std::popcount(p[0]) + std::popcount(p[1]);
    } else {
        return std::popcount((uint64_t)x);
    }
}

static inline int ctz128(ubit_t x) {
    if constexpr (sizeof(ubit_t) == 16) {
        const uint64_t *p = (const uint64_t *)&x;
        if (p[0] != 0)
            return std::countr_zero(p[0]);
        return 64 + std::countr_zero(p[1]);
    } else {
        return std::countr_zero((uint64_t)x);
    }
}

std::atomic<double> tsymm, tgemm;

template <int16_t L, int16_t XL> struct SymmFactor {
    size_t data[L * XL];
    constexpr SymmFactor() {
        for (int16_t x = 0; x < XL; x += 4) {
            data[(x + 0) * L] = 1;
            data[(x + 1) * L] = 1;
            data[(x + 2) * L] = 1;
            data[(x + 3) * L] = 1;
            for (int16_t jt = 1; jt < L; jt++) {
                data[(x + 0) * L + jt] = data[(x + 0) * L + jt - 1] * ((x + 0) - jt + 1) / jt;
                data[(x + 1) * L + jt] = data[(x + 1) * L + jt - 1] * ((x + 1) - jt + 1) / jt;
                data[(x + 2) * L + jt] = data[(x + 2) * L + jt - 1] * ((x + 2) - jt + 1) / jt;
                data[(x + 3) * L + jt] = data[(x + 3) * L + jt - 1] * ((x + 3) - jt + 1) / jt;
            }
        }
    }
};

struct ThreadPartitioning {
    const int16_t n_threads;
    const size_t len_m, len_n, len_k;
    static constexpr int16_t m_ratio = 1, n_ratio = 2, nt_n2_max = 3;
    std::array<int16_t, 3> factors;
    ThreadPartitioning(size_t n_threads, size_t len_m, size_t len_n, size_t len_k)
        : n_threads(n_threads), len_m(len_m), len_n(len_n), len_k(len_k) {
        factors[1] = biparition(n_threads, len_m * m_ratio, len_n * n_ratio);
        factors[0] = n_threads / factors[1];
        for (int16_t p = nt_n2_max; p > 1 && factors[0] >= 16; p--)
            if (factors[0] % p == 0) {
                factors[0] /= p;
                break;
            }
        factors[2] = n_threads / (factors[0] * factors[1]);
    }
    static int16_t biparition(int16_t n_threads, size_t len_m, size_t len_n) {
        const size_t x2n = len_m * n_threads;
        int16_t r = x2n >= n_threads * n_threads * len_n / 4 ? n_threads : 1, s = r;
        for (int16_t p = 1; p <= n_threads && p * p * len_n <= x2n; p++)
            r = n_threads % p ? r : p;
        for (int16_t p = n_threads; p > 0 && p * p * len_n > x2n; p--)
            s = n_threads % p ? s : p;
        return (r * r * len_n > x2n ? r * r * len_n - x2n : x2n - r * r * len_n) >
                       (s * s * len_n > x2n ? s * s * len_n - x2n : x2n - s * s * len_n)
                   ? s
                   : r;
    }
    static int16_t &get_n_threads(int16_t x = -1) {
        static int16_t _n_threads = 0;
        if (x >= 0)
            _n_threads = x;
        if (_n_threads == 0) {
            const char *str = std::getenv("HAST_NUM_THREADS");
            if (!str)
                str = std::getenv("OMP_NUM_THREADS");
            if (str)
                _n_threads = std::strtol(str, NULL, 10);
            else
                _n_threads = std::thread::hardware_concurrency();
        }
        return _n_threads;
    }
};

template <uint8_t MaxL> struct LayeredThreading {
    static constexpr uint32_t spin = 2048;
    struct alignas(128) pair_t {
        std::array<std::pair<std::atomic<uint32_t>, std::atomic<uint32_t>>, (MaxL + 7) / 8 * 8> vals;
        std::pair<std::atomic<uint32_t>, std::atomic<uint32_t>> &operator[](size_t idx) { return vals[idx]; }
    };
    std::array<uint32_t, MaxL> n_threads;
    std::array<uint32_t, MaxL + 1> post_prods;
    pair_t *nwps;
    void *data;
    LayeredThreading(const std::array<uint32_t, MaxL> &n_threads) : n_threads(n_threads) {
        post_prods[MaxL] = 1;
        for (uint8_t i = MaxL; i > 0; i--)
            post_prods[i - 1] = post_prods[i] * n_threads[i - 1];
        data = std::aligned_alloc(64, sizeof(pair_t) * post_prods[0] / post_prods[MaxL - 1]);
        nwps = new (data) pair_t[post_prods[0] / post_prods[MaxL - 1]];
    }
    template <uint8_t L, uint8_t M = 1> uint32_t thread_id(uint32_t tid) const {
        return tid % post_prods[L] / post_prods[L + M];
    }
    template <uint8_t L> void barrier(uint32_t tid) {
        const uint32_t expected = post_prods[L];
        if (expected == 1)
            return;
        std::atomic<uint32_t> &nwait = nwps[tid / post_prods[L]][L].first;
        std::atomic<uint32_t> &phase = nwps[tid / post_prods[L]][L].second;
        const uint32_t old = phase.load(std::memory_order_relaxed);
        if (nwait.fetch_add(1, std::memory_order_acq_rel) == expected - 1) {
            nwait.store(0, std::memory_order_relaxed);
            phase.fetch_add(1, std::memory_order_release);
        } else
            for (;;) {
                if (phase.load(std::memory_order_acquire) != old)
                    return;
                for (uint32_t n = 1; n < spin; n <<= 1) {
                    for (uint32_t i = 0; i < n; i++)
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86) || defined(__riscv)
                        asm volatile("pause" ::: "memory");
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
                        asm volatile("yield" ::: "memory");
#else
                        asm volatile("" ::: "memory");
#endif
                    if (phase.load(std::memory_order_acquire) != old)
                        return;
                }
            }
    }
    ~LayeredThreading() { std::free(data); }
    template <typename F> void parallel_run(F &&f) {
        std::thread *threads = new std::thread[post_prods[0]];
        for (uint32_t ip = 1; ip < post_prods[0]; ip++)
            threads[ip] = std::move(std::thread(f, ip, this));
        f(0, this);
        for (uint32_t ip = 1; ip < post_prods[0]; ip++)
            threads[ip].join();
        delete[] threads;
    }
};

template <typename FL, int16_t L = 36, int16_t SL = 72> struct Tensordot {
    static constexpr size_t part_n = 9984, part_k = 512, part_m = 512, kp_n = 192;
    static constexpr inline SymmFactor<L, sizeof(ubit_t) * 8> prex = {};
    int16_t ndim_a, ndim_b, ndim_c, nctr;
    std::array<size_t, L> shape_a, shape_b, shape_c;
    std::array<size_t, L> strides_a, strides_b, strides_c;
    std::array<int16_t, SL> symm_desc_a, symm_desc_b, symm_desc_c;
    std::array<int16_t, L> idx_a, idx_b, tr_c;
    const FL *a, *b;
    FL *c;
    FL alpha, beta, symm_factor;
    int16_t ndim_m, ndim_n, ndim_k;
    std::array<size_t, L> shape_m, shape_n, shape_k;
    std::array<size_t, L> strides_ma, strides_nb, strides_mc, strides_nc, strides_ka, strides_kb;
    std::array<int16_t, SL> symm_ma, symm_ka, symm_nb, symm_kb, symm_mc, symm_nc;
    size_t len_m, len_n, len_k, len_c;
    Tensordot(int16_t ndim_a, int16_t ndim_b, const size_t *shape_a, const size_t *shape_b, const size_t *strides_a,
              const size_t *strides_b, int16_t nctr, const int16_t *idx_a, const int16_t *idx_b, const int16_t *tr_c,
              const FL *a, const FL *b, FL *c, FL alpha, FL beta, const size_t *strides_c,
              const int16_t *symm_desc_a = nullptr, const int16_t *symm_desc_b = nullptr,
              const int16_t *symm_desc_c = nullptr)
        : ndim_a(ndim_a), ndim_b(ndim_b), nctr(nctr), ndim_c(ndim_a + ndim_b - nctr - nctr), a(a), b(b), c(c),
          alpha(alpha), beta(beta), symm_factor((FL)1.0) {
        assert(ndim_a <= L && ndim_b <= L && ndim_c <= L);
        std::memcpy(&this->shape_a[0], &shape_a[0], sizeof(size_t) * ndim_a);
        std::memcpy(&this->shape_b[0], &shape_b[0], sizeof(size_t) * ndim_b);
        std::memcpy(&this->strides_a[0], &strides_a[0], sizeof(size_t) * ndim_a);
        std::memcpy(&this->strides_b[0], &strides_b[0], sizeof(size_t) * ndim_b);
        std::memcpy(&this->strides_c[0], &strides_c[0], sizeof(size_t) * ndim_c);
        std::memcpy(&this->idx_a[0], &idx_a[0], sizeof(int16_t) * nctr);
        std::memcpy(&this->idx_b[0], &idx_b[0], sizeof(int16_t) * nctr);
        std::memcpy(&this->tr_c[0], &tr_c[0], sizeof(int16_t) * ndim_c);
        this->symm_desc_a[0] = 0, this->symm_desc_b[0] = 0, this->symm_desc_c[0] = 0;
        std::memcpy(&this->symm_desc_a[0], &symm_desc_a[0], sizeof(int16_t) * get_symm_desc_data_length(symm_desc_a));
        std::memcpy(&this->symm_desc_b[0], &symm_desc_b[0], sizeof(int16_t) * get_symm_desc_data_length(symm_desc_b));
        std::memcpy(&this->symm_desc_c[0], &symm_desc_c[0], sizeof(int16_t) * get_symm_desc_data_length(symm_desc_c));
        fill_c_shapes();
        fill_mnk_shapes();
    }
    static int16_t get_symm_desc_data_length(const int16_t *symm) noexcept {
        if (symm == nullptr)
            return 0;
        int16_t ip = 0;
        for (; symm[ip] != 0; ip += 1 + symm[ip])
            ;
        return ip + 1;
    }
    void fill_c_shapes() {
        std::array<int16_t, L> pidxa = {0}, pidxb = {0}, inv_tr_c;
        for (int16_t x = 0; x < nctr; x++)
            pidxa[idx_a[x]] = 1, pidxb[idx_b[x]] = 1;
        for (int16_t x = 0; x < ndim_c; x++)
            inv_tr_c[tr_c[x]] = x;
        for (int16_t x = 0, y = 0; y < ndim_a - nctr; x++)
            if (!pidxa[x])
                this->shape_c[inv_tr_c[y++]] = shape_a[x];
        for (int16_t x = 0, y = 0; y < ndim_b - nctr; x++)
            if (!pidxb[x])
                this->shape_c[inv_tr_c[ndim_a - nctr + y++]] = shape_b[x];
    }
    static void fill_symm_descs(int16_t ndim_m, int16_t ndim_k, const int16_t *pidx, const int16_t *symm_desc,
                                int16_t *__restrict__ symm_m, int16_t *__restrict__ symm_k) noexcept {
        int16_t ipm = 0, ipk = 0;
        for (int16_t ip = 0; symm_desc[ip] != 0; ip += 1 + symm_desc[ip]) {
            symm_m[ipm] = symm_k[ipk] = 0;
            for (int16_t i = 0; i < symm_desc[ip]; i++)
                if (pidx[symm_desc[ip + i + 1]] == 0)
                    symm_m[ipm]++, symm_m[ipm + symm_m[ipm]] = symm_desc[ip + i + 1];
                else
                    symm_k[ipk]++, symm_k[ipk + symm_k[ipk]] = symm_desc[ip + i + 1];
            ipm += symm_m[ipm] == 0 ? 0 : 1 + symm_m[ipm];
            ipk += symm_k[ipk] == 0 ? 0 : 1 + symm_k[ipk];
        }
        symm_m[ipm] = symm_k[ipk] = 0;
    }
    static void map_symm_idx_forward(int16_t ndim_k, const int16_t *idx, int16_t *__restrict__ symm_k) noexcept {
        for (int16_t ip = 0; symm_k[ip] != 0; ip += 1 + symm_k[ip])
            for (int16_t i = 0, j; i < symm_k[ip]; i++) {
                for (j = 0; j < ndim_k && idx[j] != symm_k[ip + i + 1]; j++)
                    ;
                symm_k[ip + i + 1] = j;
            }
    }
    static void map_symm_idx_backward(int16_t ndim_k, const int16_t *idx, int16_t *__restrict__ symm_k) noexcept {
        for (int16_t ip = 0; symm_k[ip] != 0; ip += 1 + symm_k[ip])
            for (int16_t i = 0, j; i < symm_k[ip]; i++)
                symm_k[ip + i + 1] = idx[symm_k[ip + i + 1]];
    }
    static void unify_symms(int16_t ndim_k, int16_t *__restrict__ symm_a, int16_t *__restrict__ symm_b) noexcept {
        std::array<int16_t, L> istat = {0};
        std::array<int16_t, SL> new_symm;
        int16_t iw = 0;
        for (int16_t k = 0, x; k < ndim_k;) {
            for (x = 0; x < ndim_k && istat[x] != 0; x++)
                ;
            istat[x] = 1;
            for (bool p = true, q, r; p;) {
                p = false;
                for (int16_t j = 0; j < 2; j++) {
                    int16_t *xsymm = j == 0 ? symm_a : symm_b;
                    for (int16_t ip = 0; xsymm[ip] != 0; ip += 1 + xsymm[ip]) {
                        q = false, r = false;
                        for (int16_t i = 0; i < xsymm[ip]; i++)
                            q |= (istat[xsymm[ip + 1 + i]] == 1), r |= (istat[xsymm[ip + 1 + i]] == 0);
                        p |= q && r;
                        for (int16_t i = 0; q && r && i < xsymm[ip]; i++)
                            istat[xsymm[ip + 1 + i]] = 1;
                    }
                }
            }
            std::array<size_t, 2> fx = {1, 1};
            for (int16_t j = 0; j < 2; j++) {
                int16_t *xsymm = j == 0 ? symm_a : symm_b;
                for (int16_t ip = 0; xsymm[ip] != 0; ip += 1 + xsymm[ip]) {
                    if (istat[xsymm[ip + 1]] == 1)
                        for (int16_t i = 2; i <= xsymm[ip]; i++)
                            fx[j] *= (size_t)i;
                }
            }
            int16_t *zsymm = fx[0] > fx[1] ? symm_a : symm_b;
            for (int16_t ip = 0; zsymm[ip] != 0; ip += 1 + zsymm[ip])
                if (istat[zsymm[ip + 1]] == 1) {
                    new_symm[iw] = zsymm[ip];
                    for (int16_t i = 1; i <= zsymm[ip]; i++)
                        new_symm[iw + i] = zsymm[ip + i], istat[zsymm[ip + i]] = 2;
                    k += new_symm[iw], iw += 1 + new_symm[iw];
                }
        }
        new_symm[iw] = 0;
        std::memcpy(&symm_a[0], &new_symm[0], sizeof(int16_t) * (iw + 1));
        std::memcpy(&symm_b[0], &new_symm[0], sizeof(int16_t) * (iw + 1));
    }
    static size_t get_symm_factor(const int16_t *__restrict__ symm_k) noexcept {
        size_t ctr_factor = 1;
        for (int16_t ip = 0; symm_k[ip] != 0; ip += 1 + symm_k[ip])
            for (int16_t i = 2; i <= symm_k[ip]; i++)
                ctr_factor *= (size_t)i;
        return ctr_factor;
    }
    static size_t get_symm_lens(const size_t *shape_k, const int16_t *__restrict__ symm_k) noexcept {
        size_t r = 1;
        for (int16_t ip = 0, it; symm_k[ip] != 0; ip += 1 + symm_k[ip])
            for (it = 1; it <= symm_k[ip]; it++)
                r = r * (shape_k[symm_k[ip + it]] - it + 1) / it;
        return r;
    }
    static void reorder_symm(int16_t nsm, int16_t *__restrict__ symm, const int16_t *__restrict__ perm) noexcept {
        int16_t jp = 0, im = 0, xsymm[SL];
        std::memcpy(&xsymm[0], &symm[0], sizeof(int16_t) * get_symm_desc_data_length(symm));
        for (; im < nsm; im++, jp += 1 + symm[jp])
            for (int16_t ip = 0, jm = 0; xsymm[ip] != 0; jm++, ip += 1 + xsymm[ip])
                if (jm == perm[im]) {
                    symm[jp] = xsymm[ip];
                    for (int16_t it = 1; it <= xsymm[ip]; it++)
                        symm[jp + it] = xsymm[ip + it];
                }
        symm[jp] = 0;
    }
    static void reorder_symm_with_strides(int16_t ndim_m, int16_t *__restrict__ symm_c, int16_t *__restrict__ symm_a,
                                          const size_t *__restrict__ strides_c,
                                          const size_t *__restrict__ strides_a) noexcept {
        size_t min_str[L];
        int16_t nsm = 0, perm[L];
        for (int16_t ip = 0, it; symm_c[ip] != 0; nsm++, ip += 1 + symm_c[ip]) {
            min_str[nsm] =
                strides_c[symm_c[ip + 1]] == 1
                    ? 0
                    : (strides_a != nullptr && strides_a[symm_a[ip + 1]] == 1 ? 1 : strides_c[symm_c[ip + 1]]);
            for (it = 2; it <= symm_c[ip]; it++)
                min_str[nsm] = std::min(
                    min_str[nsm],
                    strides_c[symm_c[ip + it]] == 1
                        ? 0
                        : (strides_a != nullptr && strides_a[symm_a[ip + it]] == 1 ? 1 : strides_c[symm_c[ip + it]]));
        }
        for (int16_t i = 0; i < nsm; i++)
            perm[i] = i;
        std::sort(&perm[0], &perm[nsm], [&min_str](int16_t i, int16_t j) { return min_str[i] > min_str[j]; });
        reorder_symm(nsm, &symm_c[0], &perm[0]);
        reorder_symm(nsm, &symm_a[0], &perm[0]);
    }
    void fill_mnk_shapes() {
        std::array<int16_t, L> pidxa = {0}, pidxb = {0}, pidxc = {0};
        std::array<int16_t, L> out_idx_a, out_idx_b, midx_c, nidx_c, inv_tr_c;
        ndim_m = ndim_a - nctr;
        ndim_n = ndim_b - nctr;
        ndim_k = nctr;
        symm_factor = (FL)1.0;
        for (int16_t x = 0; x < nctr; x++)
            pidxa[idx_a[x]] = 1, pidxb[idx_b[x]] = 1;
        for (int16_t x = 0; x < ndim_c; x++)
            pidxc[x] = tr_c[x] >= ndim_m;
        for (int16_t x = 0; x < ndim_c; x++)
            inv_tr_c[tr_c[x]] = x;
        for (int16_t x = 0, y = 0; y < ndim_m; x++)
            if (!pidxa[x])
                midx_c[y] = inv_tr_c[y], out_idx_a[y++] = x;
        for (int16_t x = 0, y = 0; y < ndim_n; x++)
            if (!pidxb[x])
                nidx_c[y] = inv_tr_c[y + ndim_m], out_idx_b[y++] = x;
        for (int16_t y = 0; y < ndim_m; y++) {
            shape_m[y] = shape_a[out_idx_a[y]];
            strides_ma[y] = strides_a[out_idx_a[y]];
            strides_mc[y] = strides_c[y];
        }
        for (int16_t y = 0; y < ndim_n; y++) {
            shape_n[y] = shape_b[out_idx_b[y]];
            strides_nb[y] = strides_b[out_idx_b[y]];
            strides_nc[y] = strides_c[y + ndim_m];
        }
        for (int16_t y = 0; y < ndim_k; y++) {
            shape_k[y] = shape_a[idx_a[y]];
            strides_ka[y] = strides_a[idx_a[y]];
            strides_kb[y] = strides_b[idx_b[y]];
        }
        fill_symm_descs(ndim_m, ndim_k, &pidxa[0], &symm_desc_a[0], &symm_ma[0], &symm_ka[0]);
        fill_symm_descs(ndim_n, ndim_k, &pidxb[0], &symm_desc_b[0], &symm_nb[0], &symm_kb[0]);
        fill_symm_descs(ndim_m, ndim_n, &pidxc[0], &symm_desc_c[0], &symm_mc[0], &symm_nc[0]);
        // m
        map_symm_idx_forward(ndim_m, &out_idx_a[0], &symm_ma[0]);
        map_symm_idx_forward(ndim_m, &midx_c[0], &symm_mc[0]);
        unify_symms(ndim_m, &symm_ma[0], &symm_mc[0]);
        map_symm_idx_backward(ndim_m, &out_idx_a[0], &symm_ma[0]);
        map_symm_idx_backward(ndim_m, &midx_c[0], &symm_mc[0]);
        reorder_symm_with_strides(ndim_m, &symm_mc[0], &symm_ma[0], &strides_c[0], &strides_a[0]);
        // n
        map_symm_idx_forward(ndim_n, &out_idx_b[0], &symm_nb[0]);
        map_symm_idx_forward(ndim_n, &nidx_c[0], &symm_nc[0]);
        unify_symms(ndim_n, &symm_nb[0], &symm_nc[0]);
        map_symm_idx_backward(ndim_n, &out_idx_b[0], &symm_nb[0]);
        map_symm_idx_backward(ndim_n, &nidx_c[0], &symm_nc[0]);
        reorder_symm_with_strides(ndim_n, &symm_nc[0], &symm_nb[0], &strides_c[0], &strides_b[0]);
        symm_factor *= (FL)get_symm_factor(&symm_mc[0]) * (FL)get_symm_factor(&symm_nc[0]);
        // k
        map_symm_idx_forward(ndim_k, &idx_a[0], &symm_ka[0]);
        map_symm_idx_forward(ndim_k, &idx_b[0], &symm_kb[0]);
        symm_factor *= (FL)get_symm_factor(&symm_ka[0]) * (FL)get_symm_factor(&symm_kb[0]);
        unify_symms(ndim_k, &symm_ka[0], &symm_kb[0]);
        symm_factor /= (FL)get_symm_factor(&symm_ka[0]);
        map_symm_idx_backward(ndim_k, &idx_a[0], &symm_ka[0]);
        map_symm_idx_backward(ndim_k, &idx_b[0], &symm_kb[0]);
        reorder_symm_with_strides(ndim_k, &symm_ka[0], &symm_kb[0], &strides_a[0], &strides_b[0]);
        symm_factor /= (FL)get_symm_factor(&symm_desc_c[0]);
        // len mnk
        len_m = get_symm_lens(&shape_a[0], &symm_ma[0]);
        len_k = get_symm_lens(&shape_a[0], &symm_ka[0]);
        len_n = get_symm_lens(&shape_b[0], &symm_nb[0]);
        len_c = get_symm_lens(&shape_c[0], &symm_desc_c[0]);
    }
    static void gemm_kernel_ref(size_t ni, size_t nj, size_t nk, size_t ldn, const FL f, const FL *__restrict__ xa,
                                FL *__restrict__ xb, FL *__restrict__ xc) noexcept {
        for (size_t i = 0; i < ni; i++)
            for (size_t j = 0; j < nj; j++) {
                FL x = 0.0;
                for (size_t k = 0; k < nk; k++)
                    x += xa[i * nk + k] * xb[j * nk + k];
                xc[i * ldn + j] = f * x;
            }
    }
    static void gemm_kernel(size_t ni, size_t nj, size_t nk, size_t ldn, const FL f, const FL *__restrict__ xa,
                            FL *__restrict__ xb, FL *__restrict__ xc) noexcept {
        constexpr size_t ki = 4, kj = 4;
        constexpr size_t kii = 4, kiii = 2, kjj = 4, kjjj = 2;
        size_t xni = ni / ki * ki;
        if (ni >= ki) {
            size_t xnj = nj / kj * kj;
            for (size_t xj = 0; xj < xnj; xj += kj) {
                for (size_t xi = 0; xi < xni; xi += ki) {
                    const FL *__restrict__ za = &xa[xi * nk];
                    const FL *__restrict__ zb = &xb[xj * nk];
                    FL *__restrict__ zc = &xc[xi * ldn + xj];
                    FL t[ki * kj] = {0};
                    for (size_t k = 0; k < nk; k++)
#pragma unroll kj
                        for (int j = 0; j < kj; j++)
#pragma unroll ki
                            for (int i = 0; i < ki; i++)
                                t[j * ki + i] += za[i * nk + k] * zb[j * nk + k];
#pragma unroll ki
                    for (int i = 0; i < ki; i++)
#pragma unroll kj
                        for (int j = 0; j < kj; j++)
                            zc[i * ldn + j] = f * t[j * ki + i];
                }
            }
            if (kj > kjj && ((nj - xnj) & kjj)) {
                const size_t xj = xnj;
                for (size_t xi = 0; xi < xni; xi += ki) {
                    const FL *__restrict__ za = &xa[xi * nk];
                    const FL *__restrict__ zb = &xb[xj * nk];
                    FL *__restrict__ zc = &xc[xi * ldn + xj];
                    FL t[ki * kjj] = {0};
                    for (size_t k = 0; k < nk; k++)
#pragma unroll kjj
                        for (int j = 0; j < kjj; j++)
#pragma unroll ki
                            for (int i = 0; i < ki; i++)
                                t[j * ki + i] += za[i * nk + k] * zb[j * nk + k];
#pragma unroll ki
                    for (int i = 0; i < ki; i++)
#pragma unroll kjj
                        for (int j = 0; j < kjj; j++)
                            zc[i * ldn + j] = f * t[j * ki + i];
                }
                xnj += kjj;
            }
            if (kj > kjjj && ((nj - xnj) & kjjj)) {
                const size_t xj = xnj;
                for (size_t xi = 0; xi < xni; xi += ki) {
                    const FL *__restrict__ za = &xa[xi * nk];
                    const FL *__restrict__ zb = &xb[xj * nk];
                    FL *__restrict__ zc = &xc[xi * ldn + xj];
                    FL t[ki * kjjj] = {0};
                    for (size_t k = 0; k < nk; k++)
#pragma unroll kjjj
                        for (int j = 0; j < kjjj; j++)
#pragma unroll ki
                            for (int i = 0; i < ki; i++)
                                t[j * ki + i] += za[i * nk + k] * zb[j * nk + k];
#pragma unroll ki
                    for (int i = 0; i < ki; i++)
#pragma unroll kjjj
                        for (int j = 0; j < kjjj; j++)
                            zc[i * ldn + j] = f * t[j * ki + i];
                }
                xnj += kjjj;
            }
            if ((nj - xnj) & 1) {
                const size_t xj = xnj;
                for (size_t xi = 0; xi < xni; xi += ki) {
                    const FL *__restrict__ za = &xa[xi * nk];
                    const FL *__restrict__ zb = &xb[xj * nk];
                    FL *__restrict__ zc = &xc[xi * ldn + xj];
                    FL t[ki] = {0};
                    for (size_t k = 0; k < nk; k++)
#pragma unroll ki
                        for (int i = 0; i < ki; i++)
                            t[0 + i] += za[i * nk + k] * zb[0 * nk + k];
#pragma unroll ki
                    for (int i = 0; i < ki; i++)
                        zc[i * ldn + 0] = f * t[0 + i];
                }
                xnj += 1;
            }
        }
        if (ki > kii && ((ni - xni) & kii)) {
            size_t xnj = nj / kj * kj;
            for (size_t xj = 0; xj < xnj; xj += kj) {
                const size_t xi = xni;
                const FL *__restrict__ za = &xa[xi * nk];
                const FL *__restrict__ zb = &xb[xj * nk];
                FL *__restrict__ zc = &xc[xi * ldn + xj];
                FL t[kii * kj] = {0};
                for (size_t k = 0; k < nk; k++)
#pragma unroll kj
                    for (int j = 0; j < kj; j++)
#pragma unroll kii
                        for (int i = 0; i < kii; i++)
                            t[j * kii + i] += za[i * nk + k] * zb[j * nk + k];
#pragma unroll kii
                for (int i = 0; i < kii; i++)
#pragma unroll kj
                    for (int j = 0; j < kj; j++)
                        zc[i * ldn + j] = f * t[j * kii + i];
            }
            if (kj > kjj && ((nj - xnj) & kjj)) {
                const size_t xi = xni;
                const size_t xj = xnj;
                const FL *__restrict__ za = &xa[xi * nk];
                const FL *__restrict__ zb = &xb[xj * nk];
                FL *__restrict__ zc = &xc[xi * ldn + xj];
                FL t[kii * kjj] = {0};
                for (size_t k = 0; k < nk; k++)
#pragma unroll kjj
                    for (int j = 0; j < kjj; j++)
#pragma unroll kii
                        for (int i = 0; i < kii; i++)
                            t[j * kii + i] += za[i * nk + k] * zb[j * nk + k];
#pragma unroll kii
                for (int i = 0; i < kii; i++)
#pragma unroll kjj
                    for (int j = 0; j < kjj; j++)
                        zc[i * ldn + j] = f * t[j * kii + i];
                xnj += kjj;
            }
            if (kj > kjjj && ((nj - xnj) & kjjj)) {
                const size_t xi = xni;
                const size_t xj = xnj;
                const FL *__restrict__ za = &xa[xi * nk];
                const FL *__restrict__ zb = &xb[xj * nk];
                FL *__restrict__ zc = &xc[xi * ldn + xj];
                FL t[kii * kjjj] = {0};
                for (size_t k = 0; k < nk; k++)
#pragma unroll kjjj
                    for (int j = 0; j < kjjj; j++)
#pragma unroll kii
                        for (int i = 0; i < kii; i++)
                            t[j * kii + i] += za[i * nk + k] * zb[j * nk + k];
#pragma unroll kii
                for (int i = 0; i < kii; i++)
#pragma unroll kjjj
                    for (int j = 0; j < kjjj; j++)
                        zc[i * ldn + j] = f * t[j * kii + i];
                xnj += kjjj;
            }
            if ((nj - xnj) & 1) {
                const size_t xi = xni;
                const size_t xj = xnj;
                const FL *__restrict__ za = &xa[xi * nk];
                const FL *__restrict__ zb = &xb[xj * nk];
                FL *__restrict__ zc = &xc[xi * ldn + xj];
                FL t[kii] = {0};
                for (size_t k = 0; k < nk; k++)
#pragma unroll kii
                    for (int i = 0; i < kii; i++)
                        t[0 + i] += za[i * nk + k] * zb[0 * nk + k];
#pragma unroll kii
                for (int i = 0; i < kii; i++)
                    zc[i * ldn + 0] = f * t[0 + i];
                xnj += 1;
            }
            xni += kii;
        }
        if (ki > kiii && ((ni - xni) & kiii)) {
            size_t xnj = nj / kj * kj;
            for (size_t xj = 0; xj < xnj; xj += kj) {
                const size_t xi = xni;
                const FL *__restrict__ za = &xa[xi * nk];
                const FL *__restrict__ zb = &xb[xj * nk];
                FL *__restrict__ zc = &xc[xi * ldn + xj];
                FL t[kiii * kj] = {0};
                for (size_t k = 0; k < nk; k++)
#pragma unroll kj
                    for (int j = 0; j < kj; j++)
#pragma unroll kiii
                        for (int i = 0; i < kiii; i++)
                            t[j * kiii + i] += za[i * nk + k] * zb[j * nk + k];
#pragma unroll kiii
                for (int i = 0; i < kiii; i++)
#pragma unroll kj
                    for (int j = 0; j < kj; j++)
                        zc[i * ldn + j] = f * t[j * kiii + i];
            }
            if (kj > kjj && ((nj - xnj) & kjj)) {
                const size_t xi = xni;
                const size_t xj = xnj;
                const FL *__restrict__ za = &xa[xi * nk];
                const FL *__restrict__ zb = &xb[xj * nk];
                FL *__restrict__ zc = &xc[xi * ldn + xj];
                FL t[kiii * kjj] = {0};
                for (size_t k = 0; k < nk; k++)
#pragma unroll kjj
                    for (int j = 0; j < kjj; j++)
#pragma unroll kiii
                        for (int i = 0; i < kiii; i++)
                            t[j * kiii + i] += za[i * nk + k] * zb[j * nk + k];
#pragma unroll kiii
                for (int i = 0; i < kiii; i++)
#pragma unroll kjj
                    for (int j = 0; j < kjj; j++)
                        zc[i * ldn + j] = f * t[j * kiii + i];
                xnj += kjj;
            }
            if (kj > kjjj && ((nj - xnj) & kjjj)) {
                const size_t xi = xni;
                const size_t xj = xnj;
                const FL *__restrict__ za = &xa[xi * nk];
                const FL *__restrict__ zb = &xb[xj * nk];
                FL *__restrict__ zc = &xc[xi * ldn + xj];
                FL t[kiii * kjjj] = {0};
                for (size_t k = 0; k < nk; k++)
#pragma unroll kjjj
                    for (int j = 0; j < kjjj; j++)
#pragma unroll kiii
                        for (int i = 0; i < kiii; i++)
                            t[j * kiii + i] += za[i * nk + k] * zb[j * nk + k];
#pragma unroll kiii
                for (int i = 0; i < kiii; i++)
#pragma unroll kjjj
                    for (int j = 0; j < kjjj; j++)
                        zc[i * ldn + j] = f * t[j * kiii + i];
                xnj += kjjj;
            }
            if ((nj - xnj) & 1) {
                const size_t xi = xni;
                const size_t xj = xnj;
                const FL *__restrict__ za = &xa[xi * nk];
                const FL *__restrict__ zb = &xb[xj * nk];
                FL *__restrict__ zc = &xc[xi * ldn + xj];
                FL t[kiii] = {0};
                for (size_t k = 0; k < nk; k++)
#pragma unroll kiii
                    for (int i = 0; i < kiii; i++)
                        t[0 + i] += za[i * nk + k] * zb[0 * nk + k];
#pragma unroll kiii
                for (int i = 0; i < kiii; i++)
                    zc[i * ldn + 0] = f * t[0 + i];
                xnj += 1;
            }
            xni += kiii;
        }
        if ((ni - xni) & 1) {
            size_t xnj = nj / kj * kj;
            for (size_t xj = 0; xj < xnj; xj += kj) {
                const size_t xi = xni;
                const FL *__restrict__ za = &xa[xi * nk];
                const FL *__restrict__ zb = &xb[xj * nk];
                FL *__restrict__ zc = &xc[xi * ldn + xj];
                FL t[1 * kj] = {0};
                for (size_t k = 0; k < nk; k++)
#pragma unroll kj
                    for (int j = 0; j < kj; j++)
                        t[j * 1 + 0] += za[0 * nk + k] * zb[j * nk + k];
#pragma unroll kj
                for (int j = 0; j < kj; j++)
                    zc[0 * ldn + j] = f * t[j * 1 + 0];
            }
            if (kj > kjj && ((nj - xnj) & kjj)) {
                const size_t xi = xni;
                const size_t xj = xnj;
                const FL *__restrict__ za = &xa[xi * nk];
                const FL *__restrict__ zb = &xb[xj * nk];
                FL *__restrict__ zc = &xc[xi * ldn + xj];
                FL t[1 * kjj] = {0};
                for (size_t k = 0; k < nk; k++)
#pragma unroll kjj
                    for (int j = 0; j < kjj; j++)
                        t[j * 1 + 0] += za[0 * nk + k] * zb[j * nk + k];
#pragma unroll kjj
                for (int j = 0; j < kjj; j++)
                    zc[0 * ldn + j] = f * t[j * 1 + 0];
                xnj += kjj;
            }
            if (kj > kjjj && ((nj - xnj) & kjjj)) {
                const size_t xi = xni;
                const size_t xj = xnj;
                const FL *__restrict__ za = &xa[xi * nk];
                const FL *__restrict__ zb = &xb[xj * nk];
                FL *__restrict__ zc = &xc[xi * ldn + xj];
                FL t[1 * kjjj] = {0};
                for (size_t k = 0; k < nk; k++)
#pragma unroll kjjj
                    for (int j = 0; j < kjjj; j++)
                        t[j * 1 + 0] += za[0 * nk + k] * zb[j * nk + k];
#pragma unroll kjjj
                for (int j = 0; j < kjjj; j++)
                    zc[0 * ldn + j] = f * t[j * 1 + 0];
                xnj += kjjj;
            }
            if ((nj - xnj) & 1) {
                const size_t xi = xni;
                const size_t xj = xnj;
                const FL *__restrict__ za = &xa[xi * nk];
                const FL *__restrict__ zb = &xb[xj * nk];
                FL *__restrict__ zc = &xc[xi * ldn + xj];
                FL t[1] = {0};
                for (size_t k = 0; k < nk; k++)
                    t[0 + 0] += za[0 * nk + k] * zb[0 * nk + k];
                zc[0 * ldn + 0] = f * t[0 + 0];
                xnj += 1;
            }
            xni += 1;
        }
    }
    static void addr_to_bits(const int16_t xsymm, const size_t xshape, size_t addr, const int16_t *__restrict__ xpats,
                             const int16_t *__restrict__ perm, ubit_t *__restrict__ rs) noexcept {
        size_t cn, ch, xx, i;
        for (int16_t it = xsymm, jt; it > 1; it--) {
            for (cn = xshape, i = 0; cn; cn = ch) {
                for (ch = cn / 2, xx = i + ch, jt = 1; jt < it; jt++)
                    xx = xx * (i + ch - jt) / (jt + 1);
                i += xx <= addr ? cn - ch : 0;
            }
            for (xx = --i, jt = 1; jt < it; jt++)
                xx = xx * (i - jt) / (jt + 1);
            addr -= xx, rs[xpats[perm[it - 1]]] |= ((ubit_t)1 << i);
        }
        rs[xpats[perm[0]]] |= ((ubit_t)1 << addr);
    }
    static int8_t next_perm(int16_t *__restrict__ xperm, int16_t *__restrict__ pperm, const int16_t *__restrict__ xsymm,
                            const int16_t *__restrict__ xpats, const int16_t *__restrict__ ppats,
                            const int16_t *__restrict__ patperm) noexcept {

        bool ldone = false;
        int8_t fl = 1;
        for (int16_t ip = 0, it; xsymm[ip] != 0; ip += 1 + xsymm[ip]) {
            auto pat_comp = [ppats, &ip](int16_t i, int16_t j) { return ppats[ip + i] < ppats[ip + j]; };
            if (!ldone && std::next_permutation(&pperm[ip], &pperm[ip + xsymm[ip]], pat_comp))
                ldone = true;
            for (int16_t it = 0; it < xsymm[ip]; it++)
                xperm[ip + patperm[ip + it]] = patperm[ip + pperm[ip + it]];
            for (int16_t it = 0; it < xsymm[ip]; it++)
                for (int16_t jt = it + 1; jt < xsymm[ip]; jt++)
                    if (xpats[ip + xperm[ip + it]] != xpats[ip + xperm[ip + jt]] && xperm[ip + it] > xperm[ip + jt])
                        fl = -fl;
        }
        return ldone ? fl : (int8_t)0;
    }
    static size_t symm_map_perm_count(const int16_t *__restrict__ symm, const int16_t *__restrict__ xsymm) noexcept {
        int16_t xpats[L], pat_cnt[L];
        for (int16_t ip = 0, iq = 0, it; symm[ip] != 0; iq++, ip += 1 + symm[ip])
            for (it = 1; it <= symm[ip]; it++)
                xpats[symm[ip + it]] = iq;
        size_t xperm_cnt = 1;
        for (int16_t ip = 0, it; xsymm[ip] != 0; ip += 1 + xsymm[ip]) {
            for (it = 1; it <= xsymm[ip]; it++)
                pat_cnt[xpats[xsymm[ip + it]]] = 0;
            for (it = 1; it <= xsymm[ip]; it++)
                xperm_cnt = xperm_cnt * it / ++pat_cnt[xpats[xsymm[ip + it]]];
        }
        return xperm_cnt;
    }
    // work size = nsymmk * nk * nkperm + nkperm / 8
    template <bool forward>
    static void symm_resolve(int16_t ndim_a, const size_t *__restrict__ shape_a, const size_t *__restrict__ strides_a,
                             const FL *__restrict__ ra, FL *__restrict__ rb, const int16_t *__restrict__ symm,
                             const int16_t *__restrict__ lsymm, const int16_t *__restrict__ rsymm, size_t il, size_t jl,
                             size_t ir, size_t jr, ubit_t *__restrict__ work) noexcept {
        const size_t nl = jl - il, nr = jr - ir;
        int16_t nsl = 0, nsr = 0, nsx = 0;
        // new symm shape
        size_t lshape[L], rshape[L], lstrides[L], rstrides[L], laddr[L], raddr[L];
        for (int16_t ipl = 0, it; lsymm[ipl] != 0; nsl++, ipl += 1 + lsymm[ipl])
            for (it = 1, lshape[nsl] = 1; it <= lsymm[ipl]; it++)
                lshape[nsl] = lshape[nsl] * (shape_a[lsymm[ipl + it]] - it + 1) / it;
        for (int16_t ipr = 0, it; rsymm[ipr] != 0; nsr++, ipr += 1 + rsymm[ipr])
            for (it = 1, rshape[nsr] = 1; it <= rsymm[ipr]; it++)
                rshape[nsr] = rshape[nsr] * (shape_a[rsymm[ipr + it]] - it + 1) / it;
        // new symm strides
        lstrides[nsl - 1] = 1, rstrides[nsr - 1] = 1;
        for (int16_t ix = nsl - 1; ix > 0; ix--)
            lstrides[ix - 1] = lstrides[ix] * lshape[ix];
        for (int16_t ix = nsr - 1; ix > 0; ix--)
            rstrides[ix - 1] = rstrides[ix] * rshape[ix];
        // for permutations for sum
        int16_t lpats[SL], rpats[SL], xpats[L], zpats[L], lperm[SL], rperm[SL], lcnt[L] = {0}, rcnt[L] = {0};
        for (int16_t ip = 0, iq = 0, it; symm[ip] != 0; nsx++, iq++, ip += 1 + symm[ip])
            for (it = 1; it <= symm[ip]; it++)
                xpats[symm[ip + it]] = iq;
        for (int16_t ipl = 0, it; lsymm[ipl] != 0; ipl += 1 + lsymm[ipl])
            for (it = 0; it < lsymm[ipl]; it++)
                lpats[ipl + it] = xpats[lsymm[ipl + it + 1]], zpats[lsymm[ipl + it + 1]] = ipl + it,
                            lcnt[lpats[ipl + it]]++;
        for (int16_t ipr = 0, it; rsymm[ipr] != 0; ipr += 1 + rsymm[ipr])
            for (it = 0; it < rsymm[ipr]; it++)
                rpats[ipr + it] = xpats[rsymm[ipr + it + 1]], zpats[rsymm[ipr + it + 1]] = ipr + it + SL,
                            rcnt[rpats[ipr + it]]++;

        // compute permutations
        const size_t lperm_cnt = symm_map_perm_count(symm, lsymm);
        const size_t rperm_cnt = symm_map_perm_count(symm, rsymm);

        // prepare for right perm
        int16_t patperm[SL], ppats[SL], pperm[SL];
        for (int16_t ip = 0, it, jt; rsymm[ip] != 0; ip += 1 + rsymm[ip]) {
            for (it = 0; it < rsymm[ip]; it++)
                patperm[ip + it] = it;
            for (it = 0; it < rsymm[ip]; it++)
                for (jt = rsymm[ip] - 1; jt > it; jt--)
                    if (rpats[ip + patperm[ip + jt - 1]] > rpats[ip + patperm[ip + jt]])
                        std::swap(patperm[ip + jt - 1], patperm[ip + jt]);
            for (it = 0; it < rsymm[ip]; it++)
                ppats[ip + it] = rpats[ip + patperm[ip + it]], pperm[ip + it] = it,
                           rperm[ip + patperm[ip + it]] = patperm[ip + pperm[ip + it]];
        }

        // compute cistr
        int8_t *frs = (int8_t *)(void *)(work + nr * rperm_cnt * nsx);
        frs[0] = 1;
        for (int16_t ix = 0, ip = 0; ix < nsr; ix++, ip += 1 + rsymm[ip])
            for (int16_t it = 0; it < rsymm[ip]; it++)
                for (int16_t jt = it + 1; jt < rsymm[ip]; jt++)
                    if (rpats[ip + rperm[ip + it]] != rpats[ip + rperm[ip + jt]] && rperm[ip + it] > rperm[ip + jt])
                        frs[0] = -frs[0];
        std::memset(work, 0, nr * rperm_cnt * nsx * sizeof(ubit_t));
        for (size_t gr = 0; gr < rperm_cnt; gr++) {
            for (size_t kr = ir; kr < jr; kr++) {
                ubit_t *__restrict__ grci = &work[gr * nr * nsx + (kr - ir) * nsx];
                for (int16_t ix = 0, ipr = 0; ix < nsr; ix++, ipr += 1 + rsymm[ipr]) {
                    raddr[ix] = kr / rstrides[ix] % rshape[ix];
                    addr_to_bits(rsymm[ipr], shape_a[rsymm[ipr + 1]], raddr[ix], &rpats[ipr], &rperm[ipr], &grci[0]);
                }
                for (int16_t ix = 0; ix < nsx; ix++)
                    assert(popcount128(grci[ix]) == rcnt[ix]);
            }
            frs[gr + 1] = next_perm(&rperm[0], &pperm[0], &rsymm[0], &rpats[0], &ppats[0], &patperm[0]);
        }
        assert(frs[rperm_cnt] == 0);

        // prepare for left perm
        for (int16_t ip = 0, it, jt; lsymm[ip] != 0; ip += 1 + lsymm[ip]) {
            for (it = 0; it < lsymm[ip]; it++)
                patperm[ip + it] = it;
            for (it = 0; it < lsymm[ip]; it++)
                for (jt = lsymm[ip] - 1; jt > it; jt--)
                    if (lpats[ip + patperm[ip + jt - 1]] > lpats[ip + patperm[ip + jt]])
                        std::swap(patperm[ip + jt - 1], patperm[ip + jt]);
            for (it = 0; it < lsymm[ip]; it++)
                ppats[ip + it] = lpats[ip + patperm[ip + it]], pperm[ip + it] = it,
                           lperm[ip + patperm[ip + it]] = patperm[ip + pperm[ip + it]];
        }

        int16_t xsms[SL];
        int8_t gf = 1;
        std::memcpy(&xsms[0], &lsymm[0], SL * sizeof(int16_t));
        for (int16_t ip = 0, it, jt; xsms[ip] != 0; ip += 1 + xsms[ip])
            for (it = 1; it < xsms[ip]; it++)
                for (jt = 0; jt < xsms[ip] - it; jt++)
                    if (zpats[xsms[ip + jt + 1]] > zpats[xsms[ip + jt + 2]])
                        std::swap(xsms[ip + jt + 1], xsms[ip + jt + 2]), gf = -gf;
        std::memcpy(&xsms[0], &rsymm[0], SL * sizeof(int16_t));
        for (int16_t ip = 0, it, jt; xsms[ip] != 0; ip += 1 + xsms[ip])
            for (it = 1; it < xsms[ip]; it++)
                for (jt = 0; jt < xsms[ip] - it; jt++)
                    if (zpats[xsms[ip + jt + 1]] > zpats[xsms[ip + jt + 2]])
                        std::swap(xsms[ip + jt + 1], xsms[ip + jt + 2]), gf = -gf;
        std::memcpy(&xsms[0], &symm[0], SL * sizeof(int16_t));
        for (int16_t ip = 0, it, jt; xsms[ip] != 0; ip += 1 + xsms[ip])
            for (it = 1; it < xsms[ip]; it++)
                for (jt = 0; jt < xsms[ip] - it; jt++)
                    if (zpats[xsms[ip + jt + 1]] > zpats[xsms[ip + jt + 2]])
                        std::swap(xsms[ip + jt + 1], xsms[ip + jt + 2]), gf = -gf;
        int8_t fl = 1;
        for (int16_t ix = 0, ip = 0; ix < nsl; ix++, ip += 1 + lsymm[ip])
            for (int16_t it = 0; it < lsymm[ip]; it++)
                for (int16_t jt = it + 1; jt < lsymm[ip]; jt++)
                    if (lpats[ip + lperm[ip + it]] != lpats[ip + lperm[ip + jt]] && lperm[ip + it] > lperm[ip + jt])
                        fl = -fl;
        for (size_t gl = 0; gl < lperm_cnt; gl++) {
            for (size_t kl = il; kl < jl; kl++) {
                ubit_t glci[L] = {0}, glmk[L] = {0}, gt;
                for (int16_t ix = 0, ipl = 0; ix < nsl; ix++, ipl += 1 + lsymm[ipl]) {
                    laddr[ix] = kl / lstrides[ix] % lshape[ix];
                    addr_to_bits(lsymm[ipl], shape_a[lsymm[ipl + 1]], laddr[ix], &lpats[ipl], &lperm[ipl], &glci[0]);
                }
                for (int16_t ix = 0; ix < nsx; ix++)
                    assert(popcount128(glci[ix]) == lcnt[ix]);
                for (int16_t ix = 0; ix < nsx; ix++)
                    for (gt = glci[ix]; gt; gt &= (gt - 1))
                        glmk[ix] ^= gt - 1;
                for (size_t gr = 0; gr < rperm_cnt; gr++) {
                    int8_t fr = frs[gr];
                    const size_t xnr = (jr - ir) / 4 * 4;
                    for (size_t kr = ir; kr < ir + xnr; kr += 4) {
                        if (forward && gl == 0 && gr == 0) {
                            rb[(kl - il) * nr + (kr + 0 - ir)] = 0.0;
                            rb[(kl - il) * nr + (kr + 1 - ir)] = 0.0;
                            rb[(kl - il) * nr + (kr + 2 - ir)] = 0.0;
                            rb[(kl - il) * nr + (kr + 3 - ir)] = 0.0;
                        }
                        const ubit_t *__restrict__ grci = &work[(kr - ir) * nsx + gr * nr * nsx];
                        size_t idx[4] = {0};
                        int8_t f0 = gf, f1 = gf, f2 = gf, f3 = gf;
                        for (int16_t ip = 0, ix = 0, it; ix < nsx; ix++, ip += 1 + symm[ip]) {
                            ubit_t x0 = glci[ix] | grci[ix + nsx * 0];
                            ubit_t x1 = glci[ix] | grci[ix + nsx * 1];
                            ubit_t x2 = glci[ix] | grci[ix + nsx * 2];
                            ubit_t x3 = glci[ix] | grci[ix + nsx * 3];
                            f0 = (glci[ix] & grci[ix + nsx * 0]) ? 0 : f0;
                            f1 = (glci[ix] & grci[ix + nsx * 1]) ? 0 : f1;
                            f2 = (glci[ix] & grci[ix + nsx * 2]) ? 0 : f2;
                            f3 = (glci[ix] & grci[ix + nsx * 3]) ? 0 : f3;
                            f0 = (popcount128(grci[ix + nsx * 0] & glmk[ix]) & 1) ? -f0 : f0;
                            f1 = (popcount128(grci[ix + nsx * 1] & glmk[ix]) & 1) ? -f1 : f1;
                            f2 = (popcount128(grci[ix + nsx * 2] & glmk[ix]) & 1) ? -f2 : f2;
                            f3 = (popcount128(grci[ix + nsx * 3] & glmk[ix]) & 1) ? -f3 : f3;
                            size_t xidx[4] = {0};
                            for (it = 1; it <= symm[ip]; it++) {
                                xidx[0] += prex.data[ctz128(x0) * L + it];
                                xidx[1] += prex.data[ctz128(x1) * L + it];
                                xidx[2] += prex.data[ctz128(x2) * L + it];
                                xidx[3] += prex.data[ctz128(x3) * L + it];
                                x0 &= (x0 - 1);
                                x1 &= (x1 - 1);
                                x2 &= (x2 - 1);
                                x3 &= (x3 - 1);
                            }
                            idx[0] += xidx[0] * strides_a[symm[ip + 1]];
                            idx[1] += xidx[1] * strides_a[symm[ip + 1]];
                            idx[2] += xidx[2] * strides_a[symm[ip + 1]];
                            idx[3] += xidx[3] * strides_a[symm[ip + 1]];
                        }
                        if (forward) {
                            if (f0 != 0)
                                rb[(kl - il) * nr + (kr - ir + 0)] += f0 * fl * fr * ra[idx[0]];
                            if (f1 != 0)
                                rb[(kl - il) * nr + (kr - ir + 1)] += f1 * fl * fr * ra[idx[1]];
                            if (f2 != 0)
                                rb[(kl - il) * nr + (kr - ir + 2)] += f2 * fl * fr * ra[idx[2]];
                            if (f3 != 0)
                                rb[(kl - il) * nr + (kr - ir + 3)] += f3 * fl * fr * ra[idx[3]];
                        } else {
                            if (f0 != 0)
                                std::atomic_ref<FL>(rb[idx[0]])
                                    .fetch_add(f0 * fl * fr * ra[(kl - il) * nr + (kr - ir + 0)],
                                               std::memory_order_relaxed);
                            if (f1 != 0)
                                std::atomic_ref<FL>(rb[idx[1]])
                                    .fetch_add(f1 * fl * fr * ra[(kl - il) * nr + (kr - ir + 1)],
                                               std::memory_order_relaxed);
                            if (f2 != 0)
                                std::atomic_ref<FL>(rb[idx[2]])
                                    .fetch_add(f2 * fl * fr * ra[(kl - il) * nr + (kr - ir + 2)],
                                               std::memory_order_relaxed);
                            if (f3 != 0)
                                std::atomic_ref<FL>(rb[idx[3]])
                                    .fetch_add(f3 * fl * fr * ra[(kl - il) * nr + (kr - ir + 3)],
                                               std::memory_order_relaxed);
                        }
                    }
                    for (size_t kr = ir + xnr; kr < jr; kr++) {
                        if (forward && gl == 0 && gr == 0)
                            rb[(kl - il) * nr + (kr - ir)] = 0.0;
                        const ubit_t *__restrict__ grci = &work[(kr - ir) * nsx + gr * nr * nsx];
                        size_t idx = 0, xidx;
                        int8_t f = gf;
                        for (int16_t ip = 0, ix = 0, it; ix < nsx; ix++, ip += 1 + symm[ip]) {
                            ubit_t x = glci[ix] | grci[ix];
                            f = (glci[ix] & grci[ix]) ? 0 : f;
                            if (f == 0)
                                break;
                            f = (popcount128(grci[ix] & glmk[ix]) & 1) ? -f : f;
                            for (xidx = 0, it = 1; x; it++, x &= (x - 1))
                                xidx += prex.data[ctz128(x) * L + it];
                            idx += xidx * strides_a[symm[ip + 1]];
                        }
                        if (f != 0) {
                            if (forward)
                                rb[(kl - il) * nr + (kr - ir)] += f * fl * fr * ra[idx];
                            else
                                std::atomic_ref<FL>(rb[idx]).fetch_add(f * fl * fr * ra[(kl - il) * nr + (kr - ir)],
                                                                       std::memory_order_relaxed);
                        }
                    }
                }
            }
            fl = next_perm(&lperm[0], &pperm[0], &lsymm[0], &lpats[0], &ppats[0], &patperm[0]);
        }
        assert(fl == 0);
    }
    std::array<double, 5> compute() noexcept {
        constexpr size_t n_align = 256;
        const size_t lp_m = std::min(len_m, part_m), lp_n = std::min(len_n, part_n);
        const size_t lp_n2 = std::min(len_n, kp_n), lp_k = std::min(len_k, part_k);
        const size_t nsm =
            std::max(get_symm_desc_data_length(&symm_desc_a[0]),
                     std::max(get_symm_desc_data_length(&symm_desc_b[0]), get_symm_desc_data_length(&symm_desc_c[0])));
        const size_t mpc = std::max(symm_map_perm_count(&symm_desc_a[0], &symm_ka[0]),
                                    std::max(symm_map_perm_count(&symm_desc_b[0], &symm_kb[0]),
                                             symm_map_perm_count(&symm_desc_c[0], &symm_nc[0])));

        const size_t pa_z = (lp_m * lp_k + n_align - 1) / n_align * n_align;
        const size_t pb_z = (lp_n * lp_k + n_align - 1) / n_align * n_align;
        const size_t pc_z = (lp_m * lp_n2 + n_align - 1) / n_align * n_align;
        const size_t psm_z = (std::max(lp_k, lp_n2) * nsm * mpc + (mpc + 7) / 8 + n_align - 1) / n_align * n_align;
        FL *__restrict__ pack_a = (FL *)std::aligned_alloc(n_align, pa_z * sizeof(FL));
        FL *__restrict__ pack_b = (FL *)std::aligned_alloc(n_align, pb_z * sizeof(FL));
        FL *__restrict__ pack_c = (FL *)std::aligned_alloc(n_align, pc_z * sizeof(FL));
        ubit_t *__restrict__ sm_work = (ubit_t *)std::aligned_alloc(n_align, psm_z * sizeof(ubit_t));
        FL f = symm_factor * alpha;

        if (beta == 0.0)
            std::memset(c, 0, len_c * sizeof(FL));
        else if (beta != 1.0)
            for (size_t i = 0; i < len_c; i++)
                c[i] *= beta;
        Timer _t;
        _t.get_time();

        size_t cur_pn = len_n, cur_pk = len_k, cur_pm = len_m;
        for (size_t idx_pn = 0; idx_pn < len_n; idx_pn += part_n) {
            const size_t len_pn = std::min(part_n, len_n - idx_pn);
            for (size_t idx_pk = 0; idx_pk < len_k; idx_pk += part_k) {
                const size_t len_pk = std::min(part_k, len_k - idx_pk);
                for (size_t idx_pm = 0; idx_pm < len_m; idx_pm += part_m) {
                    const size_t len_pm = std::min(part_m, len_m - idx_pm);
                    _t.get_time();
                    if (cur_pn != idx_pn || cur_pk != idx_pk)
                        symm_resolve<true>(ndim_b, &shape_b[0], &strides_b[0], b, pack_b, &symm_desc_b[0], &symm_nb[0],
                                           &symm_kb[0], idx_pn, idx_pn + len_pn, idx_pk, idx_pk + len_pk, sm_work);
                    if (cur_pm != idx_pm || cur_pk != idx_pk)
                        symm_resolve<true>(ndim_a, &shape_a[0], &strides_a[0], a, pack_a, &symm_desc_a[0], &symm_ma[0],
                                           &symm_ka[0], idx_pm, idx_pm + len_pm, idx_pk, idx_pk + len_pk, sm_work);
                    tsymm.fetch_add(_t.get_time());
                    cur_pn = idx_pn, cur_pk = idx_pk, cur_pm = idx_pm;
                    for (size_t idx_kpn = 0; idx_kpn < len_pn; idx_kpn += kp_n) {
                        const size_t len_kpn = std::min(kp_n, len_pn - idx_kpn);
                        _t.get_time();
                        gemm_kernel(len_pm, len_kpn, len_pk, len_kpn, f, pack_a, pack_b + idx_kpn * len_pk, pack_c);
                        tgemm.fetch_add(_t.get_time());
                        symm_resolve<false>(ndim_c, &shape_c[0], &strides_c[0], pack_c, c, &symm_desc_c[0], &symm_mc[0],
                                            &symm_nc[0], idx_pm, idx_pm + len_pm, idx_pn + idx_kpn,
                                            idx_pn + idx_kpn + len_kpn, sm_work);
                        tsymm.fetch_add(_t.get_time());
                    }
                }
            }
        }
        std::free(pack_a), std::free(pack_b), std::free(pack_c), std::free(sm_work);
        return std::array<double, 5>{};
    }
    std::array<double, 5> parallel_compute(int16_t n_threads) noexcept {
        ThreadPartitioning gt(n_threads, len_m, len_n, len_k);
        const int16_t np_n = gt.factors[0], np_m = gt.factors[1], np_n2 = gt.factors[2];
        constexpr size_t ker_kn = 4, ker_n = 4, ker_m = 4, ker_k = 4, ker_c = 4;
        const int16_t np_nm = np_n * np_m, np_mn2 = np_m * np_n2;
        constexpr size_t n_align = 64;
        const size_t lp_m = std::min(len_m, part_m), lp_n = std::min(len_n, part_n);
        const size_t lp_n2 = std::min(len_n, kp_n), lp_k = std::min(len_k, part_k);
        const size_t nsm =
            std::max(get_symm_desc_data_length(&symm_desc_a[0]),
                     std::max(get_symm_desc_data_length(&symm_desc_b[0]), get_symm_desc_data_length(&symm_desc_c[0])));
        const size_t mpc = std::max(symm_map_perm_count(&symm_desc_a[0], &symm_ka[0]),
                                    std::max(symm_map_perm_count(&symm_desc_b[0], &symm_kb[0]),
                                             symm_map_perm_count(&symm_desc_c[0], &symm_nc[0])));
        const size_t pa_z = (lp_m * lp_k + n_align - 1) / n_align * n_align;
        const size_t pb_z = (lp_n * lp_k + n_align - 1) / n_align * n_align;
        const size_t pc_z = (lp_m * lp_n2 + n_align - 1) / n_align * n_align;
        const size_t psm_z = (std::max(lp_k, lp_n2) * nsm * mpc + (mpc + 7) / 8 + n_align - 1) / n_align * n_align;
        FL *__restrict__ pack_a = (FL *)std::aligned_alloc(n_align, pa_z * np_n * np_m * sizeof(FL));
        FL *__restrict__ pack_b = (FL *)std::aligned_alloc(n_align, pb_z * np_n * sizeof(FL));
        FL *__restrict__ pack_c = (FL *)std::aligned_alloc(n_align, pc_z * np_n * np_m * np_n2 * sizeof(FL));
        ubit_t *__restrict__ sm_work = (ubit_t *)std::aligned_alloc(n_align, psm_z * n_threads * sizeof(ubit_t));
        FL f = symm_factor * alpha;

        LayeredThreading<3>({(uint32_t)np_n, (uint32_t)np_m, (uint32_t)np_n2})
            .parallel_run([&](uint32_t thread_id, LayeredThreading<3> *layered) {
                Timer _t;
                _t.get_time();
                const size_t p_c = ((len_c + n_threads - 1) / n_threads + ker_c - 1) / ker_c * ker_c;
                const size_t pst_c = thread_id * p_c, ped_c = std::min(len_c, (thread_id + 1) * p_c);
                if (beta == 0.0) {
                    if (ped_c > pst_c)
                        std::memset(c + pst_c, 0, (ped_c - pst_c) * sizeof(FL));
                } else if (beta != 1.0)
                    for (size_t idx_pc = pst_c; idx_pc < ped_c; idx_pc++)
                        c[idx_pc] *= beta;

                layered->barrier<0>(thread_id);

                const int16_t pidx_n = layered->thread_id<0>(thread_id);
                const size_t p_n = ((len_n + np_n - 1) / np_n + ker_kn - 1) / ker_kn * ker_kn;
                const size_t pst_n = pidx_n * p_n, ped_n = std::min(len_n, (pidx_n + 1) * p_n);

                for (size_t idx_pn = pst_n; idx_pn < ped_n; idx_pn += part_n) {
                    const size_t len_pn = std::min(part_n, ped_n - idx_pn);
                    const int16_t pidx_mn2 = layered->thread_id<1, 2>(thread_id);
                    const size_t pp_n = ((len_pn + np_mn2 - 1) / np_mn2 + ker_n - 1) / ker_n * ker_n;
                    const size_t ppst_n = pidx_mn2 * pp_n, pped_n = std::min(len_pn, (pidx_mn2 + 1) * pp_n);

                    for (size_t idx_pk = 0; idx_pk < len_k; idx_pk += part_k) {
                        const size_t len_pk = std::min(part_k, len_k - idx_pk);
                        const size_t pp_k = ((len_pk + np_mn2 - 1) / np_mn2 + ker_k - 1) / ker_k * ker_k;
                        const size_t ppst_k = pidx_mn2 * pp_k, pped_k = std::min(len_pk, (pidx_mn2 + 1) * pp_k);

                        layered->barrier<1>(thread_id);

                        _t.get_time();
                        if (pped_n > ppst_n)
                            symm_resolve<true>(ndim_b, &shape_b[0], &strides_b[0], b,
                                               &pack_b[pb_z * pidx_n + ppst_n * len_pk], &symm_desc_b[0], &symm_nb[0],
                                               &symm_kb[0], idx_pn + ppst_n, idx_pn + pped_n, idx_pk, idx_pk + len_pk,
                                               &sm_work[thread_id * psm_z]);
                        tsymm.fetch_add(_t.get_time());

                        layered->barrier<1>(thread_id);

                        const int16_t pidx_m = layered->thread_id<1>(thread_id);
                        const int16_t pidx_nm = pidx_m + pidx_n * np_m;
                        const size_t p_m = ((len_m + np_m - 1) / np_m + ker_m - 1) / ker_m * ker_m;
                        const size_t pst_m = pidx_m * p_m, ped_m = std::min(len_m, (pidx_m + 1) * p_m);

                        for (size_t idx_pm = pst_m; idx_pm < ped_m; idx_pm += part_m) {
                            const size_t len_pm = std::min(part_m, ped_m - idx_pm);

                            layered->barrier<2>(thread_id);

                            const int16_t pidx_n2 = layered->thread_id<2>(thread_id);
                            const int16_t pidx_nmn2 = pidx_n2 + pidx_nm * np_n2;
                            const size_t pp_m = ((len_pm + np_n2 - 1) / np_n2 + ker_m - 1) / ker_m * ker_m;
                            const size_t ppst_m = pidx_n2 * pp_m, pped_m = std::min(len_pm, (pidx_n2 + 1) * pp_m);

                            _t.get_time();
                            if (pped_m > ppst_m)
                                symm_resolve<true>(ndim_a, &shape_a[0], &strides_a[0], a,
                                                   &pack_a[pa_z * pidx_nm + ppst_m * len_pk], &symm_desc_a[0],
                                                   &symm_ma[0], &symm_ka[0], idx_pm + ppst_m, idx_pm + pped_m, idx_pk,
                                                   idx_pk + len_pk, &sm_work[thread_id * psm_z]);
                            tsymm.fetch_add(_t.get_time());

                            layered->barrier<2>(thread_id);

                            const size_t p_n2 = ((len_pn + np_n2 - 1) / np_n2 + ker_kn - 1) / ker_kn * ker_kn;
                            const size_t pst_n2 = pidx_n2 * p_n2, ped_n2 = std::min(len_pn, (pidx_n2 + 1) * p_n2);

                            for (size_t idx_kpn = pst_n2; idx_kpn < ped_n2; idx_kpn += kp_n) {
                                const size_t len_kpn = std::min(kp_n, ped_n2 - idx_kpn);
                                _t.get_time();
                                gemm_kernel(len_pm, len_kpn, len_pk, len_kpn, f, &pack_a[pa_z * pidx_nm],
                                            &pack_b[pb_z * pidx_n + idx_kpn * len_pk], &pack_c[pc_z * pidx_nmn2]);
                                tgemm.fetch_add(_t.get_time());
                                symm_resolve<false>(ndim_c, &shape_c[0], &strides_c[0], &pack_c[pc_z * pidx_nmn2], c,
                                                    &symm_desc_c[0], &symm_mc[0], &symm_nc[0], idx_pm, idx_pm + len_pm,
                                                    idx_pn + idx_kpn, idx_pn + idx_kpn + len_kpn,
                                                    &sm_work[thread_id * psm_z]);
                                tsymm.fetch_add(_t.get_time());
                            }
                        }
                        layered->barrier<1>(thread_id);
                    }
                }
            });

        std::free(pack_a), std::free(pack_b), std::free(pack_c), std::free(sm_work);
        return std::array<double, 5>{};
    }
    template <typename T> static std::string array_to_str(int16_t n, const T &data, int16_t width = 10) noexcept {
        std::stringstream ss;
        for (int16_t i = 0; i < n; i++)
            ss << std::setw(width) << (ssize_t)data[i] << " ";
        return ss.str();
    }
    template <typename T> static std::string symm_to_str(const T &symm) noexcept {
        std::stringstream ss;
        for (int16_t ip = 0; symm[ip] != 0; ip += 1 + symm[ip]) {
            ss << "[" << symm[ip] << "] ";
            for (int16_t i = 1; i <= symm[ip]; i++)
                ss << symm[ip + i] << " ";
        }
        std::string r = ss.str();
        return r.length() != 0 ? r.substr(0, r.length() - 1) : r;
    }
    std::string to_str() const {
        std::stringstream ss;
        ss << "NDIMA = " << ndim_a << " NDIMB = " << ndim_b << " NCTR = " << nctr << " ";
        ss << "ALPHA = " << alpha << " BETA = " << beta << std::endl;
        ss << "SYMM F = " << symm_factor << std::endl;
        ss << "SHAPE A   = " << array_to_str(ndim_a, shape_a) << std::endl;
        ss << "STRIDES A = " << array_to_str(ndim_a, strides_a) << std::endl;
        ss << "SHAPE B   = " << array_to_str(ndim_b, shape_b) << std::endl;
        ss << "STRIDES B = " << array_to_str(ndim_b, strides_b) << std::endl;
        ss << "SHAPE C   = " << array_to_str(ndim_c, shape_c) << std::endl;
        ss << "STRIDES C = " << array_to_str(ndim_c, strides_c) << std::endl;
        ss << "SYMM DESC A = " << symm_to_str(symm_desc_a) << std::endl;
        ss << "SYMM DESC B = " << symm_to_str(symm_desc_b) << std::endl;
        ss << "SYMM DESC C = " << symm_to_str(symm_desc_c) << std::endl;
        ss << "IDX A = " << array_to_str(nctr, idx_a) << std::endl;
        ss << "IDX B = " << array_to_str(nctr, idx_b) << std::endl;
        ss << "TR C  = " << array_to_str(ndim_c, tr_c) << std::endl;
        ss << "NDIM M = " << ndim_m << " N = " << ndim_n << " K = " << ndim_k << " ";
        ss << "LEN M = " << len_m << " N = " << len_n << " K = " << len_k << std::endl;
        ss << "SHAPE M    = " << array_to_str(ndim_m, shape_m) << std::endl;
        ss << "STRIDES MA = " << array_to_str(ndim_m, strides_ma) << std::endl;
        ss << "STRIDES MC = " << array_to_str(ndim_m, strides_mc) << std::endl;
        ss << "SYMM DESC MA = " << symm_to_str(symm_ma) << std::endl;
        ss << "SYMM DESC MC = " << symm_to_str(symm_mc) << std::endl;
        ss << "SHAPE N    = " << array_to_str(ndim_n, shape_n) << std::endl;
        ss << "STRIDES NB = " << array_to_str(ndim_n, strides_nb) << std::endl;
        ss << "STRIDES NC = " << array_to_str(ndim_n, strides_nc) << std::endl;
        ss << "SYMM DESC NB = " << symm_to_str(symm_nb) << std::endl;
        ss << "SYMM DESC NC = " << symm_to_str(symm_nc) << std::endl;
        ss << "SHAPE K    = " << array_to_str(ndim_k, shape_k) << std::endl;
        ss << "STRIDES KA = " << array_to_str(ndim_k, strides_ka) << std::endl;
        ss << "STRIDES KB = " << array_to_str(ndim_k, strides_kb) << std::endl;
        ss << "SYMM DESC KA = " << symm_to_str(symm_ka) << std::endl;
        ss << "SYMM DESC KB = " << symm_to_str(symm_kb) << std::endl;
        return ss.str();
    }
    friend std::ostream &operator<<(std::ostream &os, const Tensordot &x) {
        os << x.to_str();
        return os;
    }
};

template <typename FL, int16_t L = 36, int16_t SL = 72> struct Transpose {
    using TD = Tensordot<FL, L, SL>;
    int16_t ndim_a;
    std::array<size_t, L> shape_a;
    std::array<size_t, L> strides_a, strides_b;
    std::array<int16_t, SL> symm_desc_a, symm_desc_b;
    const FL *a;
    FL *b;
    FL alpha, beta, symm_factor;
    Transpose(int16_t ndim_a, const size_t *shape_a, const size_t *strides_a, const size_t *strides_b,
              const int16_t *tr_b, const FL *a, FL *b, FL alpha, FL beta, const int16_t *symm_desc_a,
              const int16_t *symm_desc_b)
        : ndim_a(ndim_a), a(a), b(b), alpha(alpha), beta(beta), symm_factor((FL)1.0) {
        assert(ndim_a <= L);
        std::memcpy(&this->shape_a[0], &shape_a[0], sizeof(size_t) * ndim_a);
        std::memcpy(&this->strides_a[0], &strides_a[0], sizeof(size_t) * ndim_a);
        this->symm_desc_a[0] = 0, this->symm_desc_b[0] = 0;
        std::memcpy(&this->symm_desc_a[0], &symm_desc_a[0],
                    sizeof(int16_t) * TD::get_symm_desc_data_length(symm_desc_a));
        std::memcpy(&this->symm_desc_b[0], &symm_desc_b[0],
                    sizeof(int16_t) * TD::get_symm_desc_data_length(symm_desc_b));
        for (int16_t i = 0; i < ndim_a; i++)
            this->strides_b[tr_b[i]] = strides_b[i];
        for (int16_t ip = 0, it; this->symm_desc_b[ip] != 0; ip += 1 + this->symm_desc_b[ip])
            for (it = 1; it <= this->symm_desc_b[ip]; it++)
                this->symm_desc_b[ip + it] = tr_b[this->symm_desc_b[ip + it]];
    }
    // ii jj for b range
    static void symm_resolve(int16_t ndim_a, const size_t *__restrict__ shape_a, const size_t *__restrict__ strides_a,
                             const size_t *__restrict__ strides_b, const FL *__restrict__ ra, FL *__restrict__ rb,
                             const int16_t *__restrict__ asymm, const int16_t *__restrict__ bsymm, size_t ii, size_t jj,
                             const FL alpha, const FL beta) noexcept {
        const size_t nn = jj - ii;
        const size_t xperm_cnt = TD::symm_map_perm_count(asymm, bsymm);
        int16_t xperm[SL], xsms[SL], xpats[L], bpats[SL], zpats[L], xcnt[L] = {0}, nsa = 0, nsb = 0;
        size_t xaddr[L], bshape[L], bstrides[L];
        for (int16_t ip = 0, it; bsymm[ip] != 0; nsb++, ip += 1 + bsymm[ip])
            for (it = 1, bshape[nsb] = 1; it <= bsymm[ip]; it++)
                bshape[nsb] = bshape[nsb] * (shape_a[bsymm[ip + it]] - it + 1) / it;
        bstrides[nsb - 1] = 1;
        for (int16_t ix = nsb - 1; ix > 0; ix--)
            bstrides[ix - 1] = bstrides[ix] * bshape[ix];
        for (int16_t ip = 0, iq = 0, it; asymm[ip] != 0; nsa++, iq++, ip += 1 + asymm[ip])
            for (it = 1; it <= asymm[ip]; it++)
                xpats[asymm[ip + it]] = iq;
        for (int16_t ip = 0, it; bsymm[ip] != 0; ip += 1 + bsymm[ip])
            for (it = 0; it < bsymm[ip]; it++)
                bpats[ip + it] = xpats[bsymm[ip + it + 1]], zpats[bsymm[ip + it + 1]] = ip + it, xcnt[bpats[ip + it]]++;

        // prepare for perm
        int16_t patperm[SL], ppats[SL], pperm[SL];
        for (int16_t ip = 0, it, jt; bsymm[ip] != 0; ip += 1 + bsymm[ip]) {
            for (it = 0; it < bsymm[ip]; it++)
                patperm[ip + it] = it;
            for (it = 0; it < bsymm[ip]; it++)
                for (jt = bsymm[ip] - 1; jt > it; jt--)
                    if (bpats[ip + patperm[ip + jt - 1]] > bpats[ip + patperm[ip + jt]])
                        std::swap(patperm[ip + jt - 1], patperm[ip + jt]);
            for (it = 0; it < bsymm[ip]; it++)
                ppats[ip + it] = bpats[ip + patperm[ip + it]], pperm[ip + it] = it;
            for (it = 0; it < bsymm[ip]; it++)
                xperm[ip + patperm[ip + it]] = patperm[ip + pperm[ip + it]];
        }

        int8_t gf = 1;
        std::memcpy(&xsms[0], &asymm[0], SL * sizeof(int16_t));
        for (int16_t ip = 0, it, jt; xsms[ip] != 0; ip += 1 + xsms[ip])
            for (it = 1; it < xsms[ip]; it++)
                for (jt = 0; jt < xsms[ip] - it; jt++)
                    if (zpats[xsms[ip + jt + 1]] > zpats[xsms[ip + jt + 2]])
                        std::swap(xsms[ip + jt + 1], xsms[ip + jt + 2]), gf = -gf;
        for (size_t k = ii; k < jj; k++) {
            FL rx = (FL)0.0;
            int8_t fl = 1;
            for (size_t g = 0; g < xperm_cnt; g++) {
                ubit_t gci[L] = {0};
                for (int16_t ix = 0, ip = 0; ix < nsb; ix++, ip += 1 + bsymm[ip]) {
                    xaddr[ix] = k / bstrides[ix] % bshape[ix];
                    TD::addr_to_bits(bsymm[ip], shape_a[bsymm[ip + 1]], xaddr[ix], &bpats[ip], &xperm[ip], &gci[0]);
                }
                for (int16_t ix = 0; ix < nsa; ix++)
                    assert(popcount128(gci[ix]) == xcnt[ix]);
                size_t idx = 0, xidx;
                for (int16_t ip = 0, ix = 0, it, jt; ix < nsa; ix++, ip += 1 + asymm[ip]) {
                    ubit_t x = gci[ix];
                    for (xidx = 0, it = 1; x; it++, x &= (x - 1))
                        xidx += TD::prex.data[ctz128(x) * L + it];
                    idx += xidx * strides_a[asymm[ip + 1]];
                }
                rx += gf * fl * ra[idx];
                fl = TD::next_perm(&xperm[0], &pperm[0], &bsymm[0], &bpats[0], &ppats[0], &patperm[0]);
            }
            assert(fl == 0);
            if (beta == 0.0)
                rb[k] = alpha * rx;
            else
                rb[k] = alpha * rx + beta * rb[k];
        }
    }
    std::array<double, 5> parallel_compute(int16_t n_threads) noexcept {

        const size_t len_b = TD::get_symm_lens(&shape_a[0], &symm_desc_b[0]);

        LayeredThreading<1>({(uint32_t)n_threads}).parallel_run([&](uint32_t thread_id, LayeredThreading<1> *layered) {
            const size_t p_b = (len_b + n_threads - 1) / n_threads;
            const size_t pst_b = thread_id * p_b, ped_b = std::min(len_b, (thread_id + 1) * p_b);

            if (ped_b > pst_b)
                symm_resolve(ndim_a, &shape_a[0], &strides_a[0], &strides_b[0], a, b, &symm_desc_a[0], &symm_desc_b[0],
                             pst_b, ped_b, alpha, beta);
        });

        return std::array<double, 5>{};
    }
};

extern "C" int64_t tensordot_nflops(int16_t ndim_a, int16_t ndim_b, const size_t *shape_a, const size_t *shape_b,
                                    const size_t *strides_a, const size_t *strides_b, int16_t nctr,
                                    const int16_t *idx_a, const int16_t *idx_b, const int16_t *tr_c,
                                    const size_t *strides_c, const int16_t *symm_desc_a, const int16_t *symm_desc_b,
                                    const int16_t *symm_desc_c) noexcept {
    Tensordot<double> x(ndim_a, ndim_b, shape_a, shape_b, strides_a, strides_b, nctr, idx_a, idx_b, tr_c, nullptr,
                        nullptr, nullptr, 1.0, 0.0, strides_c, symm_desc_a, symm_desc_b, symm_desc_c);
    const size_t min_m = *std::min_element(&x.strides_mc[0], &x.strides_mc[0] + x.ndim_m);
    const size_t min_n = *std::min_element(&x.strides_nc[0], &x.strides_nc[0] + x.ndim_n);
    if (min_m < min_n) {
        decltype(Tensordot<double>::tr_c) new_tr_c;
        for (int16_t i = 0; i < x.ndim_c; i++)
            new_tr_c[i] = tr_c[i] >= ndim_a - nctr ? tr_c[i] - (ndim_a - nctr) : tr_c[i] + (ndim_b - nctr);
        x = Tensordot<double>(ndim_b, ndim_a, shape_b, shape_a, strides_b, strides_a, nctr, idx_b, idx_a, &new_tr_c[0],
                              nullptr, nullptr, nullptr, 1.0, 0.0, strides_c, symm_desc_b, symm_desc_a, symm_desc_c);
    }
    return (int64_t)x.len_m * (int64_t)x.len_n * (int64_t)x.len_k;
}

extern "C" void tensordot(int16_t ndim_a, int16_t ndim_b, const size_t *shape_a, const size_t *shape_b,
                          const size_t *strides_a, const size_t *strides_b, int16_t nctr, const int16_t *idx_a,
                          const int16_t *idx_b, const int16_t *tr_c, const double *a, const double *b, double *c,
                          double alpha, double beta, const size_t *strides_c, const int16_t *symm_desc_a,
                          const int16_t *symm_desc_b, const int16_t *symm_desc_c) noexcept {
    Tensordot<double> x(ndim_a, ndim_b, shape_a, shape_b, strides_a, strides_b, nctr, idx_a, idx_b, tr_c, a, b, c,
                        alpha, beta, strides_c, symm_desc_a, symm_desc_b, symm_desc_c);
    const size_t min_m = *std::min_element(&x.strides_mc[0], &x.strides_mc[0] + x.ndim_m);
    const size_t min_n = *std::min_element(&x.strides_nc[0], &x.strides_nc[0] + x.ndim_n);
    if (min_m < min_n) {
        decltype(Tensordot<double>::tr_c) new_tr_c;
        for (int16_t i = 0; i < x.ndim_c; i++)
            new_tr_c[i] = tr_c[i] >= ndim_a - nctr ? tr_c[i] - (ndim_a - nctr) : tr_c[i] + (ndim_b - nctr);
        x = Tensordot<double>(ndim_b, ndim_a, shape_b, shape_a, strides_b, strides_a, nctr, idx_b, idx_a, &new_tr_c[0],
                              b, a, c, alpha, beta, strides_c, symm_desc_b, symm_desc_a, symm_desc_c);
    }
    auto xx = x.parallel_compute(ThreadPartitioning::get_n_threads());
}

extern "C" void transpose(int16_t ndim_a, const size_t *shape_a, const size_t *strides_a, const size_t *strides_b,
                          const int16_t *tr_b, const double *a, double *b, double alpha, double beta,
                          const int16_t *symm_desc_a, const int16_t *symm_desc_b) noexcept {
    Transpose<double> x(ndim_a, shape_a, strides_a, strides_b, tr_b, a, b, alpha, beta, symm_desc_a, symm_desc_b);
    auto xx = x.parallel_compute(ThreadPartitioning::get_n_threads());
}

extern "C" void reset_timer() noexcept {
    tsymm.store(0.0);
    tgemm.store(0.0);
}

extern "C" void check_timer(double *xtsymm, double *xtgemm) noexcept {
    *xtsymm = tsymm.load();
    *xtgemm = tgemm.load();
}
