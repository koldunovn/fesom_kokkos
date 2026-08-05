# M10 Layer-0 — derivations, cross-source table, and typo report

**Date:** 2026-08-06 · **Status:** Task 4 (derivations complete; testbed assertions land with T5–T8)
**Scope:** re-derive `cg2` (Chronopoulos–Gear), `pipecg` (Ghysels), `oati` (PIPECG-OATI) and
`pcsi` (P-CSI Chebyshev/Stiefel) from first principles, then compare against three independent
sources. **User directive: papers may contain typos — re-derive everything, triple-check,
REPORT discrepancies.** This document is the paper-appendix seed.

**Sources** (`ssh_sergey/`, gitignored):
- **[S]** Sergey Danilov's `solvers.F90` — `ssh_solve_cg2` (:3) and `ssh_solve_cg_pipelined` (:118)
- **[CV]** Cools & Vanroose, arXiv:1612.01395v3 — Alg. 4 (prec. CG-CG), Alg. 6 (prec. p-CG)
- **[P]** Huang et al., *GMD* **9**, 4209–4225 (2016) — App. B2 (ChronGear), B3 (P-CSI), C (Lanczos)
- **[T]** Tiwari & Vadhiyar, *JPDC* **163** (2022) 147–155 — Alg. 3 (PIPECG unrolled ×2), Alg. 4+5 (OATI)
- **[GV]** Golub & Varga (1961) / Saad, *Iterative Methods*, Alg. 12.1 — Chebyshev semi-iteration

Notation throughout: `A` = SSH stiffness (symmetric, verified below), `M⁻¹` = the applied
preconditioner, `r` = residual, `u = M⁻¹r`, `w = Au`, `γ = (r,u)`, `δ = (w,u)`.

---

## 0. THE HEADLINE FINDING — the preconditioner is not symmetric, and every CG-CG-family
## recurrence silently assumes that it is

### 0.1 What the code builds

`fesom_ssh_preconditioner` (`src/fesom_ssh.cpp:265-275`) builds, per row `i`:

```
pr[i,i] = 1/d_i                                     (d_i = A_ii)
pr[i,j] = −0.5 · (a_ij / d_i) / (d_i + d_j)         (j ≠ i, same sparsity as A)
```

The transpose entry carries `d_j` in the leading division, so

```
pr[i,j] / pr[j,i] = d_j / d_i         ⇒   M⁻¹ ≠ (M⁻¹)ᵀ  whenever  d_i ≠ d_j.
```

### 0.2 Measured (lab `--sym-check`, CORE2 np8 dumps step 1 and step 20, 99.3 % pair coverage)

| operator | max\|X_ij − X_ji\| | max\|X_ij\| | **defect ratio** |
|---|---|---|---|
| `pr_values` (M⁻¹) | 8.561e-08 | 1.342e-07 | **0.638** |
| `values` (A) — control | 2.150e-05 | 1.510e+08 | 1.42e-13 |

A is symmetric to rounding (the instrument is sound). **M⁻¹ is asymmetric at order unity
relative to its own off-diagonal scale** — this is a structural property, not a small
perturbation: the defect is large wherever neighbouring diagonals differ, i.e. wherever cell
area or depth changes, which on an unstructured ocean mesh is everywhere.

### 0.3 Why the baseline PCG does not care

Baseline PCG forms `α_i = γ_i / (p_i, A p_i)` with **an explicit dot product every iteration**.
Nothing is recurred, so no orthogonality identity is invoked. Empirically it converges (128.8
iters/solve on CORE2) because `M⁻¹` is still a decent approximate inverse — it just is not a
self-adjoint one in the Euclidean inner product.

### 0.4 Why cg2 / pipecg / oati DO care (derived here, first time)

The whole CG-CG family replaces that explicit `(p_i, A p_i)` with the recurrence

```
σ_i = δ_i − β_i² σ_{i-1} ,    α_i = γ_i / σ_i             (σ_i ≡ (p_i, A p_i))
```

Derivation. With `p_i = u_i + β_i p_{i-1}` and A symmetric,

```
(p_i, A p_i) = (u_i, A u_i) + 2β_i (u_i, A p_{i-1}) + β_i² (p_{i-1}, A p_{i-1})
             = δ_i + 2β_i (u_i, s_{i-1}) + β_i² σ_{i-1}          [s_{i-1} = A p_{i-1}]
```

so the recurrence `σ_i = δ_i − β_i² σ_{i-1}` holds **iff**

```
(u_i, s_{i-1}) = −β_i σ_{i-1}.                                       (★)
```

Now `s_{i-1} = (r_{i-1} − r_i)/α_{i-1}`, hence

```
(u_i, s_{i-1}) = [ (M⁻¹r_i, r_{i-1}) − γ_i ] / α_{i-1}.
```

`(M⁻¹ r_i, r_{i-1}) = (r_i, M⁻ᵀ r_{i-1})`. **If and only if `M⁻ᵀ = M⁻¹`** does this equal
`(r_i, M⁻¹ r_{i-1}) = 0` — the standard PCG residual M-orthogonality. Then

```
(u_i, s_{i-1}) = −γ_i/α_{i-1} = −β_i γ_{i-1}/α_{i-1} = −β_i σ_{i-1} ✓  = (★).
```

**With a non-symmetric `M⁻¹` the term `(r_i, M⁻ᵀ r_{i-1})` is not zero, (★) fails, and `σ_i`
drifts away from the true `(p_i, A p_i)` — cumulatively, since σ is a recurrence.** A wrong σ
gives a wrong α, which is no longer the line-search minimiser; the method stops being CG.
This applies to `cg2`, `pipecg` and `oati` alike (they share this α recurrence).

### 0.4b MEASURED — on the real CORE2 matrix, and it is not small

`tools/fesom_ssh_lab --sigma-drift` runs reference PCG on a dumped ocean system while
carrying BOTH the true `(p_i,Ap_i)` and the CG-CG recurrence `σ_i = δ_i − β_i²σ_{i-1}`, first
with the preconditioner as built and then with the symmetrised one of §0.5. CORE2 np1,
step 20 dump (`labdumps/core2_np1/step0020`, nnz 870 146), 61 iterations:

| iteration | as-built: relative σ drift | as-built: `\|(u_i,r_{i-1})\|/γ` (must be 0) | symmetrised: σ drift |
|--:|--:|--:|--:|
| 2 | 1.98e-02 | 1.28e-02 | 5.4e-14 |
| 8 | 8.20e-02 | 8.11e-02 | 8.3e-14 |
| 20 | 1.53e-01 | 1.66e-01 | 5.4e-14 |
| 40 | 1.77e-01 | 2.00e-01 | 3.8e-14 |
| 60 | **2.18e-01** | **2.41e-01** | 1.5e-14 |
| **worst** | **21.8 %** | **24.1 %** | **1.17e-13** |

**The recurred σ that `cg2`/`pipecg`/`oati` would use is wrong by up to 22 % — so their α is
wrong by up to 22 % — on the production CORE2 matrix, at the production tolerance.** The
orthogonality term that the derivation requires to vanish instead reaches 24 % of γ. With the
symmetrised preconditioner both quantities sit at machine precision (1.2e-13), i.e. the
identity holds exactly, as the derivation says it must.

This is a **quantified and now numerically CONFIRMED candidate explanation for the CG²
instability Sergey observed** — same algorithm, same matrix class, same preconditioner.
The same experiment on the 1024-unknown variable-diagonal testbed fixture reproduces it
(drift 4.2e-03 as-built vs 1.6e-14 symmetrised, `tests/test_ssh_solvers.cpp` §4); on a
CONSTANT-diagonal Laplacian the effect vanishes entirely (defect 0), which is why a uniform
model problem would never have revealed it.

The same requirement appears independently inside `oati`: its `γ_{i+1}` recurrence (`[T]`
Alg. 4 line 11) contracts `(r_i, q_i)` and `(s_i, u_i)` into one term, which is only valid
because `(r_i, M⁻¹w_i) = (M⁻¹r_i, w_i)` — again M-symmetry. And `pcsi` needs it for the
Chebyshev/Lanczos spectral theory (real spectrum, outward-converging Ritz values).

**⇒ Conclusion, and a scope change vs the plan: the preconditioner-symmetry decision that the
plan scheduled for P-CSI only (R1, Task 4 → Task 8) governs ALL FOUR solvers.**

### 0.5 The fix, and why it is nearly free

Factor the built preconditioner as

```
M⁻¹ = D⁻¹ C ,   D = diag(d_i) ,   C_ii = 1 ,   C_ij = −0.5 a_ij/(d_i + d_j)  (i≠j).
```

`C` **is symmetric** — `a_ij = a_ji` (measured 1.4e-13) and `(d_i+d_j)` is symmetric. So the
asymmetry lives entirely in the `D⁻¹` on the left. Define

```
M̃⁻¹ := D^{−1/2} C D^{−1/2}       i.e.   p̃r[i,j] = pr[i,j] · sqrt(d_i/d_j),  p̃r[i,i] = 1/d_i.
```

Two facts make this the right object:

1. **`M̃⁻¹` is symmetric** by construction (both `C` and `D^{−1/2}` are).
2. **`M̃⁻¹` is SIMILAR to `M⁻¹`**: `D^{1/2}(D⁻¹C)D^{−1/2} = D^{−1/2}CD^{−1/2}`. Similar
   matrices have **identical spectra**, so `M̃⁻¹` is exactly the same quality of approximate
   inverse — we are not trading conditioning for symmetry, we are choosing the self-adjoint
   representative of the same operator. *(Verified: `λmax(M⁻¹)` and `λmax(M̃⁻¹)` agree to
   10 significant figures on the variable-diagonal fixture, 2.008783027e-01 both.)*

   ⚠️ **The similarity is a statement about `M⁻¹`, NOT about the preconditioned operator.**
   `M̃⁻¹A` is *not* similar to `M⁻¹A` (`D^{1/2}(D⁻¹CA)D^{−1/2} = D^{−1/2}CAD^{−1/2} ≠
   M̃⁻¹A` in general), so the preconditioned spectrum does shift a little — **measured 0.23 %
   in `λmax` on the variable-diagonal fixture** (1.203422 → 1.206141). Same convergence class,
   not the identical operator; an earlier draft of this section over-claimed and the testbed
   caught it.

Positive-definiteness: `C` is strictly diagonally dominant with unit diagonal — for
`Σ_{j≠i}|a_ij| ≲ d_i` and neighbouring `d_j ≈ d_i` the off-diagonal row sum is ≈ 0.25 — hence
SPD; asserted numerically in the testbed and checked on real dumps in the lab.

Cost: one extra multiply per stored entry, applied ONCE at setup (the values are frozen —
the existing `pr_values` freeze precedent). Zero per-iteration cost. Sparsity unchanged, so the
CGPIPE ring-shipping machinery is untouched.

### 0.6 The knob (design decision, T5a)

```
FESOM_SSH_SYMPRE = 1 (default for every non-cg solver) | 0
```

`0` reproduces the literal non-symmetric MITgcm form — i.e. **Sergey's exact configuration**.
That makes §0.4 an experiment rather than an assertion: run `cg2` at `SYMPRE=0` and `SYMPRE=1`
on the same dump and compare σ-drift, α sequences and stability. Baseline `cg` NEVER consults
this knob (byte-identity), and `pcsi` refuses `SYMPRE=0` (its theory has no meaning there).

**Pre-registered prediction (written before the measurement):** at `SYMPRE=0`, the recurred σ
will drift from the true `(p,Ap)` by a relative amount that grows with iteration count and is
largest on meshes with strong depth/area contrast; at `SYMPRE=1` the drift should sit at
rounding level. If instead `SYMPRE=0` showed rounding-level drift too, §0.4 would be REFUTED.

**Outcome: CONFIRMED, and larger than expected** (§0.4b) — 21.8 % worst drift on CORE2, versus
1.2e-13 symmetrised. The prediction was directionally right and quantitatively **wrong-low**:
a >20 % error in the step length was not anticipated when the mechanism was written down.

---

## 1. `cg2` — Chronopoulos & Gear preconditioned CG

### 1.1 Own derivation (from standard PCG)

Standard PCG per iteration: `p_i = u_i + β_i p_{i-1}`; `α_i = γ_i/(p_i,Ap_i)`;
`x_{i+1} = x_i + α_i p_i`; `r_{i+1} = r_i − α_i A p_i`; `u_{i+1} = M⁻¹r_{i+1}`;
`β_{i+1} = γ_{i+1}/γ_i`.

Two substitutions remove both synchronisation points from the middle of the iteration:

- **(S)** define `s_i := A p_i`. Since `p_i = u_i + β_i p_{i-1}` and A is linear,
  `s_i = A u_i + β_i s_{i-1} = w_i + β_i s_{i-1}` with `w_i := A u_i` — the SpMV moves off `p`
  and onto `u`, and can therefore be evaluated immediately after `u`, i.e. adjacent to the dots.
- **(R)** replace `(p_i, A p_i)` by the σ recurrence of §0.4 (valid under M-symmetry):
  `σ_i = δ_i − β_i²σ_{i-1}`, `α_i = γ_i/σ_i`. Equivalently, dividing by `γ_i` and using
  `β_i = γ_i/γ_{i-1}`:

  ```
  1/α_i = δ_i/γ_i − β_i/α_{i-1}          ⇔        α_i = (δ_i/γ_i − β_i/α_{i-1})⁻¹
  ```

  which is exactly `[CV]` Alg. 4 line 13. **The two published forms are algebraically the same
  statement** — see the cross-source table.

Both `γ_{i} = (r_i,u_i)` and `δ_i = (w_i,u_i)` are now available at the same point, so **one
reduction per iteration** carries both. We add `(r_i,r_i)` for the stopping test (as `[S]`
does), giving **one fused 3-element `MPI_Allreduce`** — down from the baseline's 2 blocking
reductions (1-element + 2-element).

### 1.2 Result (the form we implement)

```
r₀ = b − Ax₀ ; u₀ = M⁻¹r₀ ; w₀ = Au₀
γ₀ = (r₀,u₀) ; δ₀ = (w₀,u₀) ; α₀ = γ₀/δ₀ ; β₀ = 0 ; p₋₁ = s₋₁ = 0
for i = 0,1,2,…
    p_i = u_i + β_i p_{i-1}
    s_i = w_i + β_i s_{i-1}
    x_{i+1} = x_i + α_i p_i
    r_{i+1} = r_i − α_i s_i
    u_{i+1} = M⁻¹ r_{i+1}                      ← preconditioner SpMV
    w_{i+1} = A u_{i+1}                        ← matrix SpMV
    ⟨γ_{i+1}, δ_{i+1}, ρ_{i+1}⟩ = ⟨(r,u), (w,u), (r,r)⟩_{i+1}   ← ONE 3-element Allreduce
    if sqrt(ρ_{i+1}/nod2D) < rtol: stop
    β_{i+1} = γ_{i+1}/γ_i
    α_{i+1} = (δ_{i+1}/γ_{i+1} − β_{i+1}/α_i)⁻¹
```

### 1.3 Ring composition (ours — not in any source)

`[S]` exchanges twice per iteration (`rr` before the M-SpMV, `uu` before the A-SpMV). Our
CGPIPE 2-ring trick removes one: exchange `r` on rings 1+2 in a single fused message; then
`u = M⁻¹r` is computable on owned **and ring-1** rows (ring-1 preconditioner rows are shipped
verbatim at setup — frozen, zstar-safe), so the subsequent `w = Au` at owned rows sees a
current ring-1 `u` with **no second message**. `p` and `s` keep their own ring-1 values by
recurrence. **⇒ 1 exchange + 1 Allreduce per iteration** (baseline: 1 exchange + 2 Allreduce).

### 1.4 Cross-source table

| item | own derivation | `[S]` `solvers.F90:3` | `[CV]` Alg. 4 | `[P]` App. B2 (ChronGear) | verdict |
|---|---|---|---|---|---|
| `s` recurrence | `s_i = w_i + β_i s_{i-1}` | :78 identical | line 6 identical | line 8 (`p_k = z_k + β_k p_{k-1}`, names swapped) | ✅ 4/4 |
| `α` recurrence | `(δ/γ − β/α_prev)⁻¹` | :102 `al=1/(sprod3(2)/sprod3(1)-be/al)` ✅ | line 13 identical | lines 4+6: `σ_k = δ_k − β_k²σ_{k-1}`, `α_k = ρ_k/σ_k` | ✅ algebraically identical (proof §0.4) |
| `β` | `γ_{i+1}/γ_i` | :101 ✅ | line 12 ✅ | line 5 ✅ | ✅ 4/4 |
| first iteration | `α₀ = γ₀/δ₀`, `β₀ = 0` | :64-65 ✅ | line 3 ✅ | `ρ₀=1, σ₀=0, s₀=p₀=0` (see T-2) | ✅ equivalent |
| reduction width | 3 (γ, δ, ρ) | 3 ✅ | 2 (no stopping dot shown) | 2 | ✅ ours = `[S]` |
| exchanges/iter | **1** (ring-composed) | 2 | n/a (not distributed) | n/a | ⭐ ours is better |
| M-symmetry required | **YES** (§0.4) | not stated | not stated | assumed ("symmetric positive definite") | ⚠️ **see T-1** |

---

## 2. `pipecg` — Ghysels/Cools–Vanroose pipelined PCG

### 2.1 Own derivation (from `cg2`)

In `cg2` the reduction (line "⟨γ,δ,ρ⟩") sits immediately **after** the two SpMVs that produce
its operands, so nothing can overlap it. Pipelining moves the SpMVs of the *next* iteration
*before* the wait:

Take `w_{i+1} = A u_{i+1} = A M⁻¹ r_{i+1}` and substitute the recurrences
`u_{i+1} = u_i − α_i q_i` (with `q_i := M⁻¹ s_i`) and `w_{i+1} = w_i − α_i z_i` (with
`z_i := A q_i = A M⁻¹ s_i`). Using `s_i = w_i + β_i s_{i-1}` and linearity,

```
q_i = M⁻¹ s_i = M⁻¹w_i + β_i q_{i-1} = m_i + β_i q_{i-1}        with  m_i := M⁻¹ w_i
z_i = A q_i   = A m_i  + β_i z_{i-1} = n_i + β_i z_{i-1}        with  n_i := A m_i
```

`m_i` and `n_i` depend only on `w_i`, which is known **before** the reduction of iteration i.
Therefore the two operator applications `m_i = M⁻¹w_i` and `n_i = Am_i` can be issued between
`MPI_Iallreduce` and `MPI_Wait` — they are the overlap payload. Everything else becomes AXPYs.

### 2.2 Result

```
r₀ = b − Ax₀ ; u₀ = M⁻¹r₀ ; w₀ = Au₀
for i = 0,1,2,…
    ⟨γ_i, δ_i, ρ_i⟩ = ⟨(r_i,u_i), (w_i,u_i), (r_i,r_i)⟩        ← Iallreduce POSTED
        m_i = M⁻¹ w_i                                          ← overlap payload
        n_i = A m_i                                            ← overlap payload
    Wait
    if sqrt(ρ_i/nod2D) < rtol: stop
    if i > 0 :  β_i = γ_i/γ_{i-1} ;  α_i = (δ_i/γ_i − β_i/α_{i-1})⁻¹
    else     :  β_i = 0           ;  α_i = γ_i/δ_i
    z_i = n_i + β_i z_{i-1} ;  q_i = m_i + β_i q_{i-1}
    s_i = w_i + β_i s_{i-1} ;  p_i = u_i + β_i p_{i-1}
    x_{i+1} = x_i + α_i p_i ;  r_{i+1} = r_i − α_i s_i
    u_{i+1} = u_i − α_i q_i ;  w_{i+1} = w_i − α_i z_i
```

4 extra vectors (`z, q, m, n`) and 8 AXPY-class updates replace `cg2`'s 4. Operator
applications per iteration are unchanged (1 × M⁻¹, 1 × A).

### 2.3 Ring composition and the R2 caveat

The overlap window contains **`w`'s exchange** (needed before `m = M⁻¹w`) — ring-composed to
one fused 2-ring message as in §1.3, so `n = Am` needs no second message.

⚠️ **The overlap is structurally unavailable on this stack.** T2's probe (jobs 26722815 /
26722816) measured `MPI_Iallreduce` progression under `MPI_THREAD_SINGLE` on both production
MPIs: wait time equals the full blocking latency at every busy-work factor, and the
nonblocking path carries a **surcharge** (+8 µs on openmpi/4.1.2, +1.6–1.8 µs on
4.1.5-nvhpc) over a blocking `Allreduce`. Pre-registered attribution (binding): a null or
negative `pipecg`-vs-`cg2` delta is **STACK**, not algorithm. This is also the most likely
explanation for the null overlap gain Sergey reported on the same machine.

### 2.4 Cross-source table

| item | own derivation | `[S]` `:118` | `[CV]` Alg. 6 | `[T]` Alg. 2/3 | verdict |
|---|---|---|---|---|---|
| aux definitions | `m=M⁻¹w`, `n=Am`, `q=M⁻¹s`, `z=Aq` | :189/:193 ✅ | lines 5,6 ✅ | ✅ | ✅ 4/4 |
| `z,q,s,p` recurrences | `·_i = ·_i + β_i ·_{i-1}` | :217-220 ✅ | 13–16 ✅ | Alg. 3 line 10-11 ✅ | ✅ 4/4 |
| `u,w` updates | `u−αq`, `w−αz` | :224-225 ✅ | 19,20 ✅ | ✅ | ✅ 4/4 |
| first-iteration branch | `i>0` test | :205 `IF(iter>1)` ✅ | 8–12 ✅ | Alg. 3 line 5 ✅ | ✅ 4/4 |
| overlap payload | `m_i`, `n_i` inside the window | :187-194 ✅ | 5,6 ✅ | ✅ | ✅ 4/4 |
| stop test placement | after Wait | :196-203 ✅ | not shown | ✅ | ✅ |
| `α` form | `(δ/γ − β/α_prev)⁻¹` | :207 ✅ | line 9 ✅ | `γ/(δ − βγ/α_prev)` | ✅ identical (multiply through by γ) |

---

## 3. `oati` — PIPECG one-allreduce-per-two-iterations

### 3.1 Own unroll (authoritative — `[T]`'s two-column listing is cross-check only)

Write two consecutive `pipecg` iterations `i`, `i+1`. The obstruction to a single reduction is
that `γ_{i+1}, δ_{i+1}` are needed to form `α_{i+1}, β_{i+1}` before iteration `i+1`'s vector
updates. OATI removes it by expressing those dots as **recurrences in already-reduced
quantities**. Expanding with `r_{i+1} = r_i − α_i s_i`, `u_{i+1} = u_i − α_i q_i`:

```
γ_{i+1} = (r_{i+1}, u_{i+1}) = γ_i − α_i[(r_i,q_i) + (s_i,u_i)] + α_i² (s_i,q_i)
```

and with `s_i = w_i + β_i s_{i-1}`, `q_i = m_i + β_i q_{i-1}`:

```
(s_i,u_i) = δ_i + β_i (u_i, s_{i-1})
(r_i,q_i) = (r_i,m_i) + β_i (r_i, q_{i-1})
(s_i,q_i) = (w_i,m_i) + β_i(w_i,q_{i-1}) + β_i(s_{i-1},m_i) + β_i²(s_{i-1},q_{i-1})
```

**⚠️ `[T]` Alg. 4 line 11 writes the `α_i` coefficient as `2α_i(λ8 + β_iλ0)`** — i.e. it
identifies `(r_i,q_i) ≡ (s_i,u_i)`. That identification requires
`(r_i, M⁻¹w_i) = (M⁻¹r_i, w_i)`, **i.e. a symmetric `M⁻¹`** (§0.4). It also folds
`(w_i,q_{i-1})` and `(s_{i-1},m_i)` into a single `2β_iλ2`-style term for the same reason.
With our `M⁻¹` as built, those foldings are invalid — a second, independent route to the same
conclusion as §0.4. Under `FESOM_SSH_SYMPRE=1` they are valid and the paper's compact form is
recovered exactly.

Same treatment for `δ_{i+1} = (w_{i+1},u_{i+1})` with `w_{i+1} = w_i − α_i z_i`:

```
δ_{i+1} = δ_i − α_i[(w_i,q_i) + (z_i,u_i)] + α_i²(z_i,q_i)
```

expanded with `z_i = n_i + β_i z_{i-1}` into `(n_i,m_i)`, `(n_i,q_{i-1})`, `(z_{i-1},q_{i-1})`
and the `λ2` family — matching `[T]` line 12 term-for-term once the symmetric-M foldings are
applied.

The new dots need `m_{i+2} = M⁻¹w_{i+2}` and `n_{i+2} = Am_{i+2}` **before** the reduction, so
those two get recurrences of their own, which introduces a deeper chain:

```
g := M⁻¹ n ,  h := A g ,  e := M⁻¹ h ,  f := A e
m_{i+1} = m_i − α_i c_i  (c_i = g_i + β_i c_{i-1})
n_{i+1} = n_i − α_i d_i  (d_i = h_i + β_i d_{i-1})
g_{i+1} = g_i − α_i a_i  (a_i = e_i + β_i a_{i-1})
h_{i+1} = h_i − α_i b_i  (b_i = f_i + β_i b_{i-1})
```

with the two "non-recurrence" repairs `a_{i+1} = (g_{i+1} − g_{i+2})/α_{i+1}`,
`b_{i+1} = (h_{i+1} − h_{i+2})/α_{i+1}` that keep the operator-application count at 2 M⁻¹ +
2 A per **combined** iteration (= 1 + 1 per single iteration, same as PCG).

### 3.2 The finding that changes OATI's expected value here

`[T]` uses **Jacobi (diagonal) preconditioning** (stated §2.2 of the paper). A diagonal `M⁻¹`
needs **no halo**. Our `M⁻¹` is sparse with A's stencil, so *every* `M⁻¹` application consumes
a halo ring exactly like an `A` application. The chain `n → g → h → e → f` is therefore
**4 operator applications deep** in our setting, and ring-composing it (no intermediate
messages) requires shipping `n` on **4 rings** per combined iteration = **2 ring-equivalents
per iteration**, against `cg2`'s 1.

> **Pre-registered expectation (T7):** `oati` halves the *reduction count* (its stated goal, and
> that part does not depend on `Iallreduce` progression) but roughly **doubles the exchanged
> halo volume** relative to `cg2`, and adds ~21 VMA-class kernel launches per combined
> iteration — priced by T2's census at **3.0 µs async / 8.9 µs fenced per launch** on A100.
> At NG5 16N the CG phase is 10.4 ms busy + 13.0 ms wait per step over ~84 iterations, so a
> saved blocking Allreduce is worth ≈ (13.0 ms / 169 reductions) ≈ 77 µs of wait, while ~10
> extra launches per iteration cost ≈ 30 µs async. The margin is real but thin, and the extra
> ring bytes eat into it. **An honest negative is paper material.**

### 3.3 Cross-source table

| item | own unroll | `[T]` Alg. 4/5 | verdict |
|---|---|---|---|
| reduction cadence | 1 per 2 iterations | ✅ line 15 | ✅ |
| reduction width | 10 (`λ0…λ9`) | ✅ 10 | ✅ (plan's "≈12" was an estimate; **10** is the number) |
| aux chain depth | `m,n,g,h,e,f` = 4 applications | ✅ line 3, 16-17 | ✅ |
| `γ_{i+1}` recurrence | §3.1 (7 terms, no folding) | folded to 5 terms | ⚠️ **T-3: folding assumes symmetric M** |
| `δ_{i+1}` recurrence | §3.1 | line 12 written as `−α_i(λ1+β_iλ2) − α_i(λ1+β_iλ2)` | ⚠️ **T-4: the same product written twice instead of `2α_i(…)`** — arithmetically identical, but it is a transcription artifact, and it hides that the two terms come from *different* inner products which are equal only under symmetric M |
| ring cost in our setting | 4 rings / combined iter | not applicable (Jacobi PC) | ⭐ our finding §3.2 |

---

## 4. `pcsi` — preconditioned Chebyshev (classical Stiefel) iteration

### 4.1 Own derivation (Golub–Varga / Saad Alg. 12.1)

Let the preconditioned operator `M⁻¹A` have spectrum in `[ν, µ]` with `0 < ν ≤ µ`. Chebyshev
semi-iteration builds `x_k` so the error is `P_k(M⁻¹A)e₀` with `P_k` the shifted-and-scaled
Chebyshev polynomial minimising the maximum on `[ν,µ]`. With

```
θ = (µ+ν)/2   (centre) ,   δ = (µ−ν)/2   (half-width) ,   σ₁ = θ/δ
```

the standard three-term form is

```
ρ₀ = 1/σ₁ ,   Δx₀ = (1/θ) M⁻¹r₀
ρ_k = 1/(2σ₁ − ρ_{k-1})
Δx_k = ρ_kρ_{k-1} Δx_{k-1} + (2ρ_k/δ) M⁻¹ r_k
```

`[P]` App. B3 uses `α = 2/(µ−ν)`, `β = (µ+ν)/(µ−ν)`, `γ = β/α`, and writes the update as
`Δx_k = ω_k r'_k + (γω_k − 1)Δx_{k-1}`. Map them:

```
α = 1/δ ,   β = θ/δ = σ₁ ,   γ = β/α = θ            ⇒ γ IS the spectrum centre
ω_k := 2ρ_k/δ
```

Check the second coefficient: `γω_k − 1 = θ(2ρ_k/δ) − 1 = 2σ₁ρ_k − 1`, and from
`ρ_k = 1/(2σ₁ − ρ_{k-1})` we have `2σ₁ρ_k − 1 = ρ_kρ_{k-1}` ✓ — the two forms agree.

### 4.2 The ω recurrence — RESOLVING the flagged ambiguity

The plan flagged `[P]` App. B3 line 1, extracted as `ω_k = 1/(γ − 1 4α2 ω_{k-1})`, as
ambiguous between `(1/4)α²` and `1/(4α²)`. Derive it:

```
ω_k = 2ρ_k/δ = 2 / (δ(2σ₁ − ρ_{k-1})) = 2 / (2θ − δρ_{k-1})
ρ_{k-1} = δω_{k-1}/2   ⇒   δρ_{k-1} = δ²ω_{k-1}/2
ω_k = 2 / (2θ − δ²ω_{k-1}/2) = 1 / (θ − (δ²/4) ω_{k-1})
```

and `δ = 1/α`, so `δ²/4 = 1/(4α²)`:

```
                    ω_k = 1 / ( γ − (1/(4α²)) · ω_{k-1} )          ✅ RESOLVED
```

**`1/(4α²)` is correct; `(1/4)α²` is wrong.** Two independent confirmations:
1. `[GV]`: the Chebyshev half-width term is `δ²/4`, and `δ = 1/α` by `[P]`'s own definition.
2. **`[P]`'s own seed is consistent only with this reading**: it sets `ω₀ = 2/γ`, and the
   standard `ρ₀ = 1/σ₁` gives `ω₀ = 2ρ₀/δ = 2/(δσ₁) = 2/θ = 2/γ` ✓. Substituting into the
   derived recurrence reproduces the textbook `ρ₁`: `ω₁ = 1/(θ − δ²/(2θ))`, and
   `2ρ₁/δ = 2/(δ(2σ₁ − 1/σ₁)) = 1/(θ − δ²/(2θ))` ✓.
   With the wrong `(1/4)α²` reading, `ω₀ = 2/γ` would not be the seed the theory requires and
   the convergence rate would be wrong — visible immediately in the lab, but silently
   *slower*, not divergent, which is exactly how such a typo survives.

Note also `[P]`'s initial step uses `Δx₀ = γ⁻¹M⁻¹r₀` (coefficient `1/θ`, **not** `ω₀`) — this
is correct and matches `[GV]`; `ω₀ = 2/γ` is a seed for the recurrence only, never a step
coefficient. A reader who assumed `Δx₀ = ω₀ M⁻¹r₀` would be off by exactly 2 on the first step.

### 4.3 Result (the form we implement)

```
γ = (µ+ν)/2 ; α = 2/(µ−ν) ; ω₀ = 2/γ
r₀ = b − Ax₀ ; Δx₀ = (1/γ)M⁻¹r₀ ; x₁ = x₀ + Δx₀ ; r₁ = b − Ax₁
for k = 1,2,…
    ω_k = 1/(γ − ω_{k-1}/(4α²))
    r'_k = M⁻¹ r_k
    Δx_k = ω_k r'_k + (γω_k − 1) Δx_{k-1}
    x_{k+1} = x_k + Δx_k
    r_{k+1} = b − A x_{k+1}                        ← TRUE residual (self-correcting)
    every FESOM_PCSI_CHECK iterations: reduce (r,r), test, else continue
```

Per iteration: 1 M⁻¹ apply, 1 A apply, 3 AXPY-class updates, **1 exchange, 0 reductions**
(one reduction every `K` iterations for the check). The residual is the *true* one by
construction, so the check-point reduction reads a residual that needs no verification — the
attainable-accuracy failure mode of pipelined methods does not exist here.

### 4.4 Requirements, and what they cost us

- **Symmetric `M⁻¹` (§0.5)** — mandatory: the Chebyshev error polynomial argument needs a real
  spectrum, and Lanczos Ritz values converge outward-short (`θmax ≤ λmax`, `θmin ≥ λmin`) only
  for a self-adjoint operator. That outward-shortness is *why* the safe margin directions are
  "deflate ν, inflate µ". `pcsi` therefore **rejects `FESOM_SSH_SYMPRE=0`**.
- **Eigenbounds** from Lanczos on `M̃⁻¹A` (`[P]` App. C, T8a). `[P]`'s listing has
  `q₁ = r₀/(r₀ᵀs₀)` — a normalisation by the **inner product, not its square root**; the
  Lanczos basis must be unit-norm in the `M⁻¹` inner product, i.e. `q₁ = r₀/sqrt(r₀ᵀs₀)`
  (**T-5**, below). With the un-rooted form the recurrence still produces a tridiagonal `T`,
  but its entries are scaled inconsistently and the Ritz values are wrong by that scale.

### 4.5 Cross-source table

| item | own derivation | `[P]` App. B3/C | `[GV]`/Saad 12.1 | verdict |
|---|---|---|---|---|
| `γ = θ` = centre | ✅ | ✅ (via β/α) | ✅ | ✅ |
| `ω` recurrence | `1/(γ − ω_{k-1}/(4α²))` | ambiguous glyph | `ρ_k = 1/(2σ₁−ρ_{k-1})` ⇒ same | ✅ **T-6 resolved: `1/(4α²)`** |
| `ω₀` seed | `2/γ` | `2/γ` ✅ | `ρ₀ = 1/σ₁` ⇒ `2/γ` ✅ | ✅ 3/3 |
| first step coefficient | `1/γ` | `γ⁻¹` ✅ | `1/θ` ✅ | ✅ 3/3 (≠ ω₀ — noted) |
| `Δx` update | `ω_k r' + (γω_k−1)Δx` | ✅ | `ρρ'Δx + (2ρ/δ)r'` ⇒ same | ✅ |
| residual | `b − Ax_{k+1}` (true) | ✅ | either | ✅ |
| Lanczos `q₁` | `r₀/sqrt(r₀ᵀs₀)` | `r₀/(r₀ᵀs₀)` | `/‖·‖` | ⚠️ **T-5** |

---

## 5. TYPO / DISCREPANCY REPORT

Severity: **S1** = wrong results if transcribed literally · **S2** = misleading or
unexecutable as written · **S3** = cosmetic.

| id | source | issue | severity | our resolution |
|---|---|---|---|---|
| **T-1** | `[CV]` Alg. 4, `[S]`, `[T]` — all CG-CG-family listings | The `α` recurrence `σ_i = δ_i − β_i²σ_{i-1}` is stated without its hypothesis. It is valid **only for a symmetric `M⁻¹`** (proof §0.4). Papers assume "SPD preconditioner" in prose; the FESOM MITgcm-class `pr_values` is **not** symmetric (defect ratio 0.638, measured). **Consequence measured on CORE2: σ — and therefore the step length α — wrong by up to 21.8 %** (§0.4b). | **S1** | `FESOM_SSH_SYMPRE=1` (default) applies `M̃⁻¹ = D^{−1/2}CD^{−1/2}` — symmetric, same preconditioner spectrum, σ-identity exact to 1.2e-13. `=0` reproduces the literal form for the falsification experiment. **Confirmed** explanation candidate for Sergey's CG² instability. |
| **T-2** | `[P]` App. B2 (ChronGear) | Line 4 computes `σ_k` using `β_k`, which is only defined on line 5 — **the listing is not executable in the order printed**. (Harmless in effect: `σ₀ = 0`, so the undefined `β₁` multiplies zero on the first pass; from k≥2 the reader must reorder.) Also `ρ₀ = 1` makes `β₁ = ρ₁` rather than 0 — again harmless only because `s₀ = p₀ = 0`. | **S2** | We compute `β` before `σ`/`α` and initialise `β₀ = 0` explicitly. |
| **T-3** | `[T]` Alg. 4 line 11 | `γ_{i+1}` recurrence folds `(r_i,q_i)` and `(s_i,u_i)` into one `2α_i(…)` term. Valid only under symmetric `M⁻¹` (same hypothesis as T-1), unstated. | **S1** (in our setting) | Own unroll keeps the terms separate; with `SYMPRE=1` the folded form is recovered and used. |
| **T-4** | `[T]` Alg. 4 line 12 | `δ_{i+1} = λ8 − α_i(λ1+β_iλ2) − α_i(λ1+β_iλ2) + …` — the identical product printed twice instead of `2α_i(…)`. Arithmetically the same, but it conceals that the two terms originate in *different* inner products, equal only under symmetric M. | **S3** (S2 as documentation) | Own unroll; noted in the paper appendix. |
| **T-5** | `[P]` App. C (Lanczos) | `q₁ = r₀/(r₀ᵀs₀)` normalises by the inner product, not its **square root**. A Lanczos start vector must be unit-norm in the working inner product. **Measured consequence** (testbed §7, 1024-unknown fixture): the missing root corrupts `α₁`, which plants a spurious tiny eigenvalue in `T`. `θmax` recovers (later vectors are renormalised correctly), but **`θmin` comes out 2.49e-06 instead of 5.65e-03 — 2270× too small, and it does NOT wash out with m** (same error at m = 30 and m = 120). Feeding that `ν` to P-CSI inflates the assumed κ by ~2000×, so the Chebyshev polynomial is optimised for a vastly-too-wide interval and the method converges far slower than it should — silently, since it still converges. | **S1** | `q₁ = r₀/sqrt(r₀ᵀs₀)`; the fixture test asserts recovery of the extreme Ritz values and that the un-rooted variant visibly fails. |
| **T-6** | `[P]` App. B3 line 1 | The `ω` recurrence coefficient is typeset as a fraction that extracts as `1 4α2`; `(1/4)α²` and `1/(4α²)` are both readable. | **S1 if misread** | **RESOLVED to `1/(4α²)`** by derivation (§4.2) + `[P]`'s own `ω₀ = 2/γ` seed + `[GV]`. Not a typo in the paper — an extraction hazard. Recorded so nobody re-litigates it. |
| **T-7** | `[T]` §2.2 vs our setting | The method is designed and benchmarked with **Jacobi (diagonal)** preconditioning, where `M⁻¹` needs no halo. Not a typo — a portability caveat that materially changes OATI's cost model with a sparse `M⁻¹` (§3.2). | **S2** (for reuse) | Ring-cost analysis §3.2; pre-registered T7 expectation. |
| **T-8** | `[S]` `solvers.F90` both routines | `MPI_Iallreduce` immediately followed by `MPI_Wait` in `ssh_solve_cg2` (:37-38, :61-62, :99-100) — a nonblocking call with a zero-length overlap window is a **blocking** reduction plus the nonblocking surcharge (T2 measured +8 µs on openmpi/4.1.2, +1.6 µs on 4.1.5). | **S3** (perf, not correctness) | We use blocking `MPI_Allreduce` in `cg2`; `Iallreduce` only in `pipecg`, where there is a real overlap window. Worth reporting to Sergey: on Levante this alone costs his `cg2` measurably. |
| **T-9** | `[S]` both routines | Hard-coded `Do iter=1,200` and no maxiter-exhaustion handling — a non-converged solve exits silently with whatever `X` holds. | **S2** | `FESOM_PCSI_MAXITER`/`FESOM_PHASE1_MAXITER` + armed fallback to baseline `cg` (T5a). |

**Nothing in `[CV]` Alg. 4 or Alg. 6 was found to be transcriptionally wrong** — those two
listings reproduce our derivations line-for-line. The substantive issues are the unstated
symmetry hypothesis (T-1/T-3, ours to fix) and the P-CSI appendix's Lanczos normalisation
(T-5).

---

## 6. What this changes in the plan

1. **`FESOM_SSH_SYMPRE` is new** (T5a), and the preconditioner decision the plan scoped to
   P-CSI (R1) now governs all four solvers. Decision: **symmetrised `D^{−1/2}CD^{−1/2}`,
   default ON for every non-`cg` solver**, chosen over a re-derived MITgcm symmetric form
   because it is provably spectrum-preserving (§0.5).
2. **`pcsi` aborts on `SYMPRE=0`** (added to the T5a interaction matrix).
3. **T5b gains a falsification leg**: `cg2` at `SYMPRE=0` vs `=1`, σ-drift and α-sequence
   compared on real dumps — the experiment that tests the CG²-instability hypothesis.
4. **T7's expectation is written down before the code** (§3.2): halved reductions, roughly
   doubled ring bytes, ~21 extra VMAs; margin thin.
5. **Reduction width for `oati` is 10, not ~12** (plan estimate corrected).
