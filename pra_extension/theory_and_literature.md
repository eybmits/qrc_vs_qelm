# Observable-restricted memory as irreducible suffix risk in quantum reservoir computing

**Standalone PRA-extension draft — 28 August 2026**
**Status:** theorem, finite-shot proposition, and bounded literature audit. This
file does not modify the frozen study or its manuscript. The audit covers the
primary sources listed below; it is not proof of absolute priority.

## One-sentence result

After a fixed input suffix, the old-prefix memory of a recurrent QRC that can
still affect the declared observables is exactly

\[
\boxed{
H(s)=\frac{1}{p}\left\|M T_s\Sigma_s^{1/2}\right\|_F^2,
}
\]

and this is exactly the minimum suffix-only mean-squared error, and hence a
lower bound for **every** deterministic model that receives only that suffix.
With finite-shot QELM features and a frozen affine readout, an additional
additive, optimally allocable measurement penalty appears under the protocol
specified below.

This is an observable-memory statement, not a full-state-memory statement. A
QRC may retain old-prefix information internally while the chosen measurement
projection is blind to it.

---

## 1. Setup in Pauli/Liouville coordinates

Let the reservoir Hilbert-space dimension be \(q=2^n\), and let

\[
P_0=I,P_1,\ldots,P_D,\qquad D=q^2-1,
\]

be the \(n\)-qubit Pauli strings, with
\(\operatorname{tr}(P_iP_j)=q\delta_{ij}\). Define the real traceless Pauli
coordinate vector of a state by

\[
x(\rho)_j=\operatorname{tr}(P_j\rho),\qquad j=1,\ldots,D.
\]

Then

\[
\rho=\frac{1}{q}\left(I+\sum_{j=1}^{D}x_j(\rho)P_j\right),
\qquad
\|\rho-\sigma\|_2^2=\frac1q\|x(\rho)-x(\sigma)\|_2^2.
\]

For an input-dependent CPTP step \(\Phi_u\), define the traceless
Pauli-transfer block and the nonunital translation by

\[
(A_u)_{ij}=\frac1q\operatorname{tr}\!\left[P_i\Phi_u(P_j)\right],
\qquad
(b_u)_i=\frac1q\operatorname{tr}\!\left[P_i\Phi_u(I)\right].
\]

The step is an affine map in these coordinates,

\[
x(\Phi_u(\rho))=A_u x(\rho)+b_u.
\]

The affine term is important for the suffix-conditioned mean. For a fixed
suffix, it cancels from state differences and conditional covariances. For a
chronological suffix

\[
s=(u_1,\ldots,u_d),
\]

define its traceless transfer product

\[
T_s=A_{u_d}A_{u_{d-1}}\cdots A_{u_1}.
\]

Thus, if \(X=x(\rho_{\mathrm{pre}})\) is the state immediately before the
suffix, the state after the suffix has coordinates

\[
x_{\mathrm{out}}=T_sX+a_s,
\]

where \(a_s\) collects the suffix translations.

Let \(p\) expectation values be reported. Their traceless action is represented
by \(M\in\mathbb R^{p\times D}\):

\[
Y=Mx_{\mathrm{out}}+c.
\]

For a selected family of Pauli targets, \(M\) is a coordinate-selection matrix
and \(MM^\top=I_p\). The formula below also covers arbitrary real linear
combinations of Pauli expectations. Identity components contribute only
constants and therefore do not affect memory.

The prefix is random under the experimental prefix ensemble. Conditional on a
fixed suffix \(s\), write

\[
\bar x_s=\mathbb E[X\mid S=s],\qquad
\Sigma_s=\operatorname{Cov}(X\mid S=s).
\]

A common-suffix design usually makes the prefix distribution independent of the
chosen suffix, but this independence is not required: the theorem uses the
conditional covariance \(\Sigma_s\).

---

## 2. Main theorem: observable suffix memory and exact prediction risk

### Theorem 1 (observable-restricted suffix-memory theorem)

Assume exact expectation-valued teacher outputs and deterministic dynamics and
readout conditional on \((X,S)\), so that
\(Y=MT_SX+(Ma_S+c)\) with no additional process or measurement randomness.
Predictors range over square-integrable measurable functions of \(S\) alone;
they receive no side information correlated with the hidden prefix. The
finite-moment assumptions below are automatic for bounded quantum observables.
Under these assumptions, fix a suffix \(s\) with positive conditional
probability, or interpret the statements for almost every \(s\) under a regular
conditional distribution. Let

\[
\mu_s=\mathbb E[Y\mid S=s].
\]

Then:

1. **Exact observable-memory covariance**

   \[
   C_s:=\operatorname{Cov}(Y\mid S=s)
   =M T_s\Sigma_sT_s^\top M^\top.
   \]

2. **Exact scalar history floor** for mean squared error averaged over the
   \(p\) reported observables

   \[
   H(s):=\frac1p\mathbb E\!\left[
   \|Y-\mu_s\|_2^2\mid S=s\right]
   =\frac1p\operatorname{tr}(C_s)
   =\frac1p\|M T_s\Sigma_s^{1/2}\|_F^2.
   \]

3. **Exact suffix-risk decomposition.** For any deterministic predictor
   \(g(s)\in\mathbb R^p\), quantum or classical,

   \[
   \frac1p\mathbb E\!\left[
   \|Y-g(s)\|_2^2\mid S=s\right]
   =H(s)+\frac1p\|\mu_s-g(s)\|_2^2.
   \]

4. **Global Bayes floor.** For a random length-\(d\) suffix \(S\),

   \[
   \mathcal R_d(g)
   :=\frac1p\mathbb E\|Y-g(S)\|_2^2
   =\mathbb E[H(S)]
   +\frac1p\mathbb E\|\mu_S-g(S)\|_2^2.
   \]

   Hence

   \[
   \inf_{g\in L^2(\sigma(S);\mathbb R^p)}\mathcal R_d(g)
   =\mathbb E[H(S)],
   \]

   with the infimum achieved by \(g^\star(s)=\mu_s\). Within this suffix-only
   information pattern, the floor is independent of the student's qubit
   number, circuit, feature map, readout nonlinearity, and training algorithm.

5. **Exact observability criterion**

   \[
   H(s)=0
   \quad\Longleftrightarrow\quad
   M T_s\Sigma_s^{1/2}=0
   \quad\Longleftrightarrow\quad
   \operatorname{ran}(\Sigma_s)\subseteq\ker(MT_s).
   \]

   Therefore zero observable memory does not imply zero state memory. It says
   only that every old-prefix direction populated by the ensemble is either
   erased by \(T_s\) or mapped into a direction invisible to \(M\).

### Proof

For a fixed suffix,

\[
Y=MT_sX+(Ma_s+c).
\]

Subtracting the conditional mean removes the deterministic translation:

\[
Y-\mu_s=MT_s(X-\bar x_s).
\]

Taking the outer product and the conditional expectation gives

\[
C_s=MT_s\Sigma_sT_s^\top M^\top.
\]

The trace identity follows from

\[
\operatorname{tr}(A\Sigma A^\top)
=\operatorname{tr}(A\Sigma^{1/2}\Sigma^{1/2}A^\top)
=\|A\Sigma^{1/2}\|_F^2
\]

with \(A=MT_s\). For any fixed \(g(s)\), expand

\[
Y-g(s)=(Y-\mu_s)+(\mu_s-g(s)).
\]

The cross term vanishes because
\(\mathbb E[Y-\mu_s\mid S=s]=0\), proving the conditional identity. Averaging
over \(S\) proves the global identity. Finally, a squared Frobenius norm is zero
if and only if its matrix argument is zero, which gives the observability
criterion. \(\square\)

If the teacher output additionally contains conditionally zero-mean noise
\(\xi\) that is uncorrelated with \(X\) given \(S\), its conditional covariance
adds to \(C_s\), and \(p^{-1}\operatorname{tr}\operatorname{Cov}(\xi\mid S=s)\)
adds to the risk floor. The noiseless theorem should not be used to relabel
teacher measurement noise as dynamical memory.

### Interpretation

The theorem separates three objects that are often conflated:

- \(T_s\Sigma_s^{1/2}\): old-prefix state directions that survive the suffix;
- \(MT_s\Sigma_s^{1/2}\): surviving directions visible in the chosen
  expectation values;
- \(\mu_s-g(s)\): the student's failure to reproduce the suffix-conditioned
  mean after the input-information bottleneck has been passed.

The history term is therefore not “the amount of memory in the quantum state.”
It is the amount of **prefix variability that survives into the declared target
observables**.

---

## 3. Exact finite-sample form for a common-suffix experiment

For one suffix and \(K>1\), let \(y_1,\ldots,y_K\) be exact expectation-valued
outputs from prefixes sampled conditionally i.i.d. given that suffix, and let

\[
\bar y=\frac1K\sum_{k=1}^{K}y_k.
\]

For any one suffix prediction \(g(s)\), the following is an exact deterministic
identity for the realized panel:

\[
\frac1{Kp}\sum_{k=1}^{K}\|y_k-g(s)\|_2^2
=
\underbrace{\frac1{Kp}\sum_{k=1}^{K}\|y_k-\bar y\|_2^2}_{\widehat H(s)}
+\frac1p\|\bar y-g(s)\|_2^2.
\]

The sample history term also has the exact pairwise form

\[
\widehat H(s)
=\frac1{K^2p}\sum_{1\leq k<\ell\leq K}\|y_k-y_\ell\|_2^2.
\]

For prefixes sampled conditionally i.i.d. given \(S=s\),

\[
\mathbb E[\widehat H(s)]=\frac{K-1}{K}H(s),
\]

so \(K\widehat H(s)/(K-1)\) is unbiased for the population floor. Equivalently,
for two conditionally independent prefix states producing \(Y,Y'\),

\[
H(s)=\frac1{2p}\mathbb E\!\left[\|Y-Y'\|_2^2
\mid S=S'=s\right].
\]

This makes the common-suffix experiment a direct estimator of the theorem's
quantity; no student needs to be trained to estimate the history floor.
The realized-panel decomposition and pairwise identity are deterministic. The
unbiased population statement additionally requires conditional i.i.d.
sampling. If each \(y_k\) is itself estimated with finite teacher shots, the
panel variance includes teacher measurement noise unless it is independently
estimated and subtracted.

---

## 4. Singular-value and covariance-eigenvalue bounds

Let

\[
A_s=MT_s,
\qquad
P_s=\text{orthogonal projector onto }\operatorname{ran}(\Sigma_s),
\qquad
\tau_s=\operatorname{tr}(\Sigma_s).
\]

If \(\Sigma_s=0\), every bound below is trivially zero. Otherwise define the
support-restricted singular values

\[
\alpha_s=\inf_{\substack{v\in\operatorname{ran}(\Sigma_s)\\\|v\|=1}}
\|A_sv\|,
\qquad
\beta_s=\sup_{\substack{v\in\operatorname{ran}(\Sigma_s)\\\|v\|=1}}
\|A_sv\|.
\]

These quantities allow \(\alpha_s=0\), as is typical when the occupied prefix
subspace has higher dimension than the measured output.

### Corollary 1 (support-restricted singular-value bounds)

\[
\boxed{
\frac{\alpha_s^2\tau_s}{p}
\leq H(s)\leq
\frac{\beta_s^2\tau_s}{p}.
}
\]

**Proof.** Diagonalize
\(\Sigma_s=\sum_i\lambda_i v_iv_i^\top\) on its support. Then

\[
pH(s)=\sum_i\lambda_i\|A_sv_i\|^2,
\]

and bound every squared norm by \(\alpha_s^2\) and \(\beta_s^2\).
\(\square\)

This is a direct history-bottleneck certificate when one can estimate the
action of the suffix on the occupied prefix subspace rather than on the entire
operator space.

Let \(\lambda_{\min}^+(\Sigma_s)\) and
\(\lambda_{\max}(\Sigma_s)\) be the smallest positive and largest covariance
eigenvalues.

### Corollary 2 (covariance-spectrum/Frobenius bounds)

\[
\boxed{
\frac{\lambda_{\min}^+(\Sigma_s)}{p}
\|A_sP_s\|_F^2
\leq H(s)\leq
\frac{\lambda_{\max}(\Sigma_s)}{p}
\|A_sP_s\|_F^2.
}
\]

This version separates the overall accessible suffix map
\(\|A_sP_s\|_F^2\) from anisotropy in the prefix ensemble.

### Corollary 3 (alignment-sensitive eigenvalue bounds)

Let \(r=\operatorname{rank}(\Sigma_s)\). On
\(\operatorname{ran}(\Sigma_s)\), let

\[
c_1\geq\cdots\geq c_r\geq0
\]

be the eigenvalues of \(P_sA_s^\top A_sP_s\), and let

\[
\lambda_1\geq\cdots\geq\lambda_r>0
\]

be the eigenvalues of \(\Sigma_s\). The von Neumann trace inequality for
positive semidefinite matrices gives

\[
\boxed{
\frac1p\sum_{i=1}^{r}c_i\lambda_{r+1-i}
\leq H(s)\leq
\frac1p\sum_{i=1}^{r}c_i\lambda_i.
}
\]

For fixed spectra \(\{c_i\}\) and \(\{\lambda_i\}\), these are extrema over
their relative eigenvector orientations: the upper extreme aligns the most
variable prefix directions with the most visible suffix modes, while the lower
extreme anti-aligns them. The realized \(H(s)\) is fixed by the actual
orientation. Thus spectra alone do not determine memory.

### Corollary 4 (measurement-subspace Ky Fan bounds)

Define the post-suffix state-memory covariance

\[
K_s=T_s\Sigma_sT_s^\top.
\]

If \(1\leq p\leq D\) and \(M\) has orthonormal rows, as for a subset of Pauli
coordinates, then
\(Q_M=M^\top M\) is a rank-\(p\) orthogonal projector and

\[
H(s)=\frac1p\operatorname{tr}(Q_MK_s).
\]

If \(k_1\geq\cdots\geq k_D\geq0\) are the eigenvalues of \(K_s\), the Ky Fan
principle yields

\[
\boxed{
\frac1p\sum_{i=D-p+1}^{D}k_i
\leq H(s)\leq
\frac1p\sum_{i=1}^{p}k_i.
}
\]

These are extrema over all rank-\(p\) linear expectation-value subspaces. They
make the role of measurement choice explicit: a \(p\)-dimensional readout can
capture at most the top-\(p\) covariance mass, and it can completely miss
memory stored in an orthogonal subspace. Arbitrary optimized linear
combinations need not correspond to one jointly measurable basis; this
corollary concerns access to exact expectation values.

---

## 5. Dynamical bounds and spectral predictions

### 5.1 Input-varying suffixes

Submultiplicativity of the operator norm gives

\[
\beta_s\leq\|M\|_2\|T_sP_s\|_2
\leq\|M\|_2\prod_{j=1}^{d}\|A_{u_j}\|_2.
\]

Therefore

\[
\boxed{
H(s)\leq
\frac{\|M\|_2^2\operatorname{tr}(\Sigma_s)}{p}
\prod_{j=1}^{d}\|A_{u_j}\|_2^2.
}
\]

If \(\|A_u\|_2\leq\kappa<1\) uniformly over the input alphabet, this becomes

\[
H(s)\leq
\frac{\|M\|_2^2\operatorname{tr}(\Sigma_s)}{p}\kappa^{2d}.
\]

This Euclidean Pauli-transfer contraction is a sufficient condition, not a
generic property of every CPTP map. A CPTP map is contractive in trace
distance, but its Hilbert--Schmidt/Pauli-coordinate operator norm need not be
strictly below one.

A sharper ensemble-aware version follows the occupied subspaces themselves.
Let \(\mathcal R_0=\operatorname{ran}(\Sigma_s)\) and
\(\mathcal R_j=A_{u_j}\mathcal R_{j-1}\), and put

\[
\kappa_j(s)=\left\|A_{u_j}|_{\mathcal R_{j-1}}\right\|_2.
\]

For a Pauli-coordinate selector, \(\|M\|_2=1\), so

\[
\boxed{
H(s)\leq\frac{\operatorname{tr}(\Sigma_s)}p
\prod_{j=1}^d\kappa_j(s)^2.
}
\]

This bound is insensitive to peripheral directions that the prefix ensemble
never occupies, while retaining any protected direction that it does occupy.
If every occupied-step gain is at most \(\bar\kappa<1\), then
\(\sqrt{H(s)}\leq\sqrt{\operatorname{tr}(\Sigma_s)/p}\,\bar\kappa^d\).

For a basis-independent alternative, let

\[
\eta(\Phi_u)=
\sup_{\rho\neq\sigma}
\frac{\|\Phi_u(\rho)-\Phi_u(\sigma)\|_1}
{\|\rho-\sigma\|_1}
\leq1
\]

be the trace-distance contraction coefficient. If output observable \(O_a\)
has operator norm \(L_a=\|O_a\|_\infty\), the independent-pair identity and
Hölder's inequality imply

\[
H(s)\leq
2\left(\frac1p\sum_{a=1}^{p}L_a^2\right)
\prod_{j=1}^{d}\eta(\Phi_{u_j})^2.
\]

For Pauli targets, \(L_a=1\). This bound is distribution independent but often
loose; \(MT_s\Sigma_s^{1/2}\) can be small even when every single-step
trace-distance coefficient equals one.

### 5.2 Repeated channel and limits of spectral-radius arguments

For a repeated input channel with traceless block \(A\), a fixed prefix
covariance \(\Sigma\), and a suffix of length \(d\),

\[
H_d=\frac1p\|MA^d\Sigma^{1/2}\|_F^2
\leq\frac{\|M\|_2^2\operatorname{tr}(\Sigma)}p\|A^d\|_2^2.
\]

If \(A=V\Lambda V^{-1}\) is diagonalizable and \(\rho(A)<1\),

\[
H_d\leq
\frac{\|M\|_2^2\operatorname{tr}(\Sigma)}p
\kappa_2(V)^2\rho(A)^{2d}.
\]

For a normal \(A\), choose a unitary \(V\), so \(\kappa_2(V)=1\). For a
defective or strongly nonnormal
map, the spectral radius alone does not control finite-window behavior:
transient amplification and Jordan polynomial factors can precede asymptotic
decay. Singular values of the actual product \(MT_sP_s\) are therefore safer
for finite \(d\).

There is a transparent special case. Suppose the normal matrix \(A\),
\(\Sigma\), and \(G=M^\top M\) are simultaneously unitarily diagonalizable,
with corresponding eigenvalues \(a_i,\lambda_i,g_i\). Then

\[
\boxed{
H_d=\frac1p\sum_i g_i\lambda_i|a_i|^{2d}.
}
\]

In this simultaneous-diagonalization case, let
\(r_{\mathrm{act}}=\max\{|a_i|:g_i\lambda_i>0\}\). The observable-amplitude
decay base is \(r_{\mathrm{act}}\), whereas the risk decay base is
\(r_{\mathrm{act}}^2\). It need not equal the full-state spectral radius.
Peripheral modes with \(|a_i|=1\) do not prevent observable forgetting when
they are unpopulated or unmeasured.

### 5.3 Heisenberg-picture reading

The exact floor can also be written

\[
H(s)=\frac1p\|\Sigma_s^{1/2}T_s^\top M^\top\|_F^2.
\]

Modulo identity components, which vanish after centering, the columns of
\(T_s^\top M^\top\) are the measured observables propagated backward through
the suffix in Pauli coordinates. Observable memory is large
only when these backward observables overlap directions in which the prefix
ensemble actually varies. This connects forgetting, observability, and the
data distribution without identifying any one of them with the others.

---

## 6. A rigorous population history-bottleneck certificate

Let

\[
V=\frac1p\mathbb E\|Y-\mathbb EY\|_2^2>0
\]

be the pooled, component-centered population target variance used to normalize
MSE, and define

\[
H_d=\mathbb E_S[H(S)].
\]

This is the population analogue of the registered NRMSE normalization. A
finite held-out NRMSE uses an empirical denominator and is not identically
equal to the population quantity below. The best achievable population
normalized RMSE of any deterministic length-\(d\) model is

\[
\operatorname{NRMSE}_{d,\star}=\sqrt{H_d/V}.
\]

Consequently, at tolerance \(\varepsilon\),

\[
H_d>\varepsilon^2V
\quad\Longrightarrow\quad
\text{no deterministic length-\(d\) model can have population NRMSE }
\leq\varepsilon.
\]

This is a distributional, observable-restricted statement. It identifies the
complete Bayes MSE floor, but it is bounded by the registered prefix/suffix
distribution, observable set, norm, and tolerance.

The singular-value bounds can certify either side without evaluating the full
trace:

- if
  \(\mathbb E[\alpha_S^2\operatorname{tr}(\Sigma_S)]/p>
  \varepsilon^2V\), history-impossibility is certified;
- if
  \(\mathbb E[\beta_S^2\operatorname{tr}(\Sigma_S)]/p\leq
  \varepsilon^2V\), the history bottleneck alone does not rule out the
  tolerance; this does not certify that a specified student family attains it.

The first statement applies equally to QELMs, kernels, neural networks, and
classical reservoirs that receive only the suffix. Therefore failure of a
well-tuned classical suffix-only control in a certified history-impossible cell
is constructive: it corroborates the information bottleneck rather than
claiming a uniquely quantum failure. In cells not excluded by the history
certificate, relative QELM/classical performance instead diagnoses
representation, estimation, or optimization.

---

## 7. Finite-shot penalty for a fixed affine QELM readout

The first theorem assumes exact expectation values. Hardware introduces a
separate bottleneck even in cells where the history floor is small.

### Proposition 2 (exact finite-shot risk increment)

Let an ideal QELM feature vector be \(z(S)\in\mathbb R^B\), with
\(Y,b\in\mathbb R^m\). Fix an affine readout

\[
f(S)=W^\top z(S)+b,\qquad W\in\mathbb R^{B\times m}.
\]

Suppose the hardware feature estimator is

\[
\widehat z(S)=z(S)+\eta,
\]

with

\[
\mathbb E[\eta\mid S,Y]=0,\qquad
\operatorname{Cov}(\eta\mid S,Y)=\Sigma_\eta(S).
\]

The covariance may depend on the input. Then the expected test MSE per output
is exactly

\[
\boxed{
\frac1m\mathbb E\left\|
Y-\left(W^\top\widehat z(S)+b\right)
\right\|_2^2
=
\frac1m\mathbb E\|Y-f(S)\|_2^2
+\frac1m\mathbb E\operatorname{tr}
\!\left(W^\top\Sigma_\eta(S)W\right).
}
\]

**Proof.** Put \(r=Y-f(S)\). The noisy error is
\(r-W^\top\eta\). Expanding its squared norm, the cross term satisfies

\[
\mathbb E[r^\top W^\top\eta]
=\mathbb E\!\left[
r^\top W^\top\mathbb E[\eta\mid S,Y]
\right]=0.
\]

For the quadratic term,

\[
\mathbb E\|W^\top\eta\|_2^2
=\mathbb E\operatorname{tr}
\!\left(W^\top\eta\eta^\top W\right)
=\mathbb E\operatorname{tr}
\!\left(W^\top\Sigma_\eta(S)W\right).
\]

This proves the identity. \(\square\)

The proposition is exact for \((W,b)\) fixed independently of the test-shot
noise. The weights may have been trained on separate data; if they are random,
the identity is read conditionally on the frozen weights and then averaged. It
is not an errors-in-variables theory for training and testing the weights on
the same noisy feature realizations.

### Corollary 5 (compatible-configuration shot allocation)

Partition the \(B\) features into \(G\) disjoint blocks. Each block corresponds
to one independently executed state-preparation/time-snapshot plus compatible
measurement-basis configuration; sharing a basis label across different
snapshots does not make those snapshots one shot block. In configuration
\(r\), a single shot returns an outcome \(R_r\) and a vector of jointly computed
estimators

\[
h_r(R_r)\in\mathbb R^{B_r},\qquad
\mathbb E[h_r(R_r)\mid S]=z_r(S).
\]

Its full single-shot covariance is

\[
C_r(S)_{ab}
=
\mathbb E[h_{r,a}(R_r)h_{r,b}(R_r)\mid S]
-z_{r,a}(S)z_{r,b}(S).
\]

If configuration \(r\) receives \(S_r\) conditionally i.i.d. shots and
distinct configurations use independent shot pools, the sample-mean feature
covariance is block diagonal:

\[
\Sigma_\eta(S)
=\bigoplus_{r=1}^{G}\frac{C_r(S)}{S_r}.
\]

Partition the corresponding rows of \(W\) as \(W_1,\ldots,W_G\), and define

\[
a_r=\frac1m\mathbb E\operatorname{tr}
\!\left(W_r^\top C_r(S)W_r\right)\geq0.
\]

The shot penalty is

\[
\mathcal P_{\mathrm{shot}}(S_1,\ldots,S_G)
=\sum_{r=1}^{G}\frac{a_r}{S_r}.
\]

Here the displayed sum assumes \(S_r>0\); zero-weight blocks can instead be
omitted. Let the active set be \(\mathcal A=\{r:a_r>0\}\). If it is nonempty,
assume a nonadaptive allocation shared across evaluated inputs, equal cost per
shot, and a continuous total-shot budget
\(\sum_{r\in\mathcal A}S_r=Q\) **per evaluated input**. Its optimum is

\[
\boxed{
S_r^\star
=Q\frac{\sqrt{a_r}}{\sum_{\ell\in\mathcal A}\sqrt{a_\ell}},
\quad r\in\mathcal A,
\qquad
\mathcal P_{\mathrm{shot}}^\star
=\frac{\left(\sum_{r\in\mathcal A}\sqrt{a_r}\right)^2}{Q}.
}
\]

Indeed, Cauchy--Schwarz gives

\[
\left(\sum_{r\in\mathcal A}\sqrt{a_r}\right)^2
=\left(\sum_{r\in\mathcal A}
\sqrt{\frac{a_r}{S_r}}\sqrt{S_r}\right)^2
\leq
\left(\sum_{r\in\mathcal A}\frac{a_r}{S_r}\right)
\left(\sum_{r\in\mathcal A}S_r\right),
\]

with equality at \(S_r\propto\sqrt{a_r}\). Configurations with \(a_r=0\) may
be omitted for this fixed readout; if all \(a_r=0\), the penalty is zero and
the allocation is immaterial. With compulsory minimum shots, the constrained
continuous optimum instead obeys the corresponding KKT/water-filling
conditions; integer shots require a discrete allocation or rounding step. The
boxed expression is only the unconstrained continuous, equal-cost solution.
For unequal per-shot costs \(c_r\) and budget \(\sum_rc_rS_r=Q\), the active
continuous solution changes to \(S_r\propto\sqrt{a_r/c_r}\).

### Why within-configuration covariance must be retained

For one state-preparation/snapshot and global Pauli-basis configuration, one
measured bitstring supplies several single-qubit and product-Pauli features at
once. Those features are generally
correlated because they are functions of the same shot. The matrices \(C_r\)
must therefore retain their off-diagonal entries. Replacing \(C_r\) by its
diagonal is justified only after an explicit covariance check; it is not
implied by Pauli compatibility.

If shot pools are shared across nominally distinct configurations,
cross-configuration covariance blocks must also be included and the simple sum
\(\sum_r a_r/S_r\) need not hold. If the allocation depends on the input,
\(S_r=S_r(S)\), its expectation and optimization must be performed with that
dependence retained. Biased mitigation, drift, teacher shot noise, and weights
fitted on noisy features require additional terms.

### Combined three-term risk law

Apply Proposition 2 to a suffix-only QELM with \(X=S\) and \(m=p\). Assume the
teacher and QELM test-shot pools are independent conditional on \(S\), the shot
estimator is conditionally unbiased even after conditioning on the hidden
teacher prefix, and \((W,b)\) is frozen independently of the test-shot noise.
Define

\[
\mathcal P_{\mathrm{shot}}
:=\frac1p\mathbb E\operatorname{tr}
\!\left(W^\top\Sigma_\eta(S)W\right).
\]

Under these assumptions, combining Proposition 2 with Theorem 1 gives the
exact population identity

\[
\boxed{
\mathcal R_{\mathrm{fixed\text{-}readout,shot}}
=
\underbrace{\mathbb E[H(S)]}_{\text{missing old history}}
+
\underbrace{\frac1p\mathbb E\|\mu_S-f(S)\|_2^2}
_{\text{ideal feature/readout residual}}
+
\underbrace{\mathcal P_{\mathrm{shot}}}_{\text{measurement budget}}.
}
\]

This is the operational phase diagram:

1. if the first term is large, no suffix-only architecture can emulate the
   teacher at the requested tolerance;
2. if the first is small but the second is large, the frozen ideal predictor
   has a large residual;
3. if the first two are small but the third is large, this affine predictor is
   measurement-limited under the specified sampling protocol.

Only when \((W,b)\) is population-optimal for the ideal frozen feature map can
the second term be interpreted purely as a feature/readout approximation
limit. For a trained finite-sample readout it can also contain estimation,
regularization, and optimization error. Likewise, the third term is a
measurement limit only for the declared frozen affine readout and measurement
protocol, not an architecture-independent hardware lower bound. The identity
also omits gate/SPAM error, drift, back-action, teacher noise, and weights
trained on the same noisy feature realizations. In the implemented fixed-
readout study, the deployed integer allocation exactly minimizes coefficients
estimated from the refit readout and train-plus-validation covariances; it is a
design-set plug-in optimum, not a test-distribution oracle.

Variance propagation through a fixed linear readout, covariance-aware grouped
measurement, and square-root resource allocation are established methods.
They are **not** claimed as new. The candidate contribution is the prospective
joint use of these ingredients with the QRC teacher's observable-history floor
on one matched canonical parallel TD-QELM emulation surface.

---

## 8. Specialization to the repository's Lindblad teacher

The canonical teacher uses

\[
\mathcal L(u)=\mathcal L_0+u\mathcal L_{\mathrm{drive}},
\qquad
\Phi_u=\exp[\Delta t\,\mathcal L(u)],
\]

with column-stacked Liouville propagation. Conjugating the full Liouville
propagator by the Pauli-basis change of coordinates and deleting the fixed
identity coordinate gives \(A_u\); the nonunital identity column gives \(b_u\).
For the frozen alphabet, every \(A_u\) can be constructed deterministically to
numerical precision once.

The theorem suggests a direct non-fit validation:

1. obtain \(K\) prefix-state Pauli vectors \(x_k\), set
   \(\bar x=K^{-1}\sum_kx_k\), and use the population-\(K\) covariance

   \[
   \widehat\Sigma_s
   =\frac1K\sum_{k=1}^{K}(x_k-\bar x)(x_k-\bar x)^\top;
   \]

2. form the exact suffix product \(T_s\);
3. select the registered target Paulis with \(M\);
4. compare

   \[
   \frac1p\operatorname{tr}
   \!\left(MT_s\widehat\Sigma_sT_s^\top M^\top\right)
   \]

   with the directly measured common-suffix history MSE;
5. define \(\widehat K_s=T_s\widehat\Sigma_sT_s^\top\) and decompose its
   surviving covariance into visible and hidden components. For a
   coordinate-selection \(M\), the hidden mass is exactly
   \(\operatorname{tr}[(I-M^\top M)\widehat K_s]\).

With identical coordinates and this denominator-\(K\) covariance, step 4 is an
algebraic equality up to floating-point error. A covariance formed with the
conventional denominator \(K-1\) is larger by \(K/(K-1)\), so its propagated
trace must be multiplied by \((K-1)/K\) to match the realized-panel history
term. Its
scientific content comes from how the spectrum and measurement overlap change
across dissipation, suffix length, and observable families. When comparing
reservoir sizes, \(q^{-1}\operatorname{tr}(\widehat K_s)\), rather than the raw
trace, is the mean-squared Hilbert--Schmidt state variation in the Pauli
normalization used here.

For hardware evaluation, record the experimental configuration blocks, the
complete within-block single-shot covariances \(C_r\), the frozen trained
readout blocks \(W_r\), and the registered per-input total budget \(Q\). The
allocation can be predicted only after \(W_r\) and \(C_r\) are available; if
the latter are unknown, obtain them from an exact simulator or registered pilot
shots before the main hardware evaluation.

---

## 9. What is established already, and what may be new

### Primary-literature audit

| Prior result | Primary source | Consequence for the claim |
|---|---|---|
| QRC was introduced as temporal processing through disordered quantum dynamics, with linear readout of measured observables. | Fujii and Nakajima, *Phys. Rev. Applied* **8**, 024030 (2017), [publisher manuscript](https://harvest.aps.org/v2/journals/articles/10.1103/PhysRevApplied.8.024030/fulltext). | Neither QRC nor measured-observable readout is new. |
| Finite-dimensional QRC admits density-matrix, observable/Bloch, and state-affine representations; echo-state and fading-memory conditions are characterized. | Martínez-Peña and Ortega, *Phys. Rev. E* **107**, 035306 (2023), [publisher](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.107.035306). | The Pauli/Bloch affine representation and QRC fading memory are not new. |
| QRC echo-state notions can be restricted to subsets/subspaces, and PTM spectral structure has been connected to fading memory. | Kobayashi, Tran, and Nakajima, *Phys. Rev. E* **110**, 024207 (2024), [publisher](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.110.024207); Kobayashi, Tran, and Nakajima, arXiv:2409.12693v2, [preprint](https://arxiv.org/abs/2409.12693). | “Only part of state space may forget” and PTM/spectral reasoning are known. Our observable-and-distribution restriction must be distinguished from their ESP definitions. |
| Contractive, input-dependent QRC filters can be analyzed for separation or injectivity of different input sequences. | Martínez-Peña and Ortega, *Phys. Rev. E* **111**, 065306 (2025), [publisher](https://journals.aps.org/pre/abstract/10.1103/3775-4hfd). | Input separation is known. “Different histories yield different states” is not a new claim. |
| Dissipative QRC families can approximate fading-memory filters under stated hypotheses. | Sannia et al., *Quantum* **8**, 1291 (2024), [article](https://quantum-journal.org/papers/q-2024-03-20-1291/). | Dissipation-assisted memory control and universality are known; the present result is finite-resource and target-observable specific. |
| Quantum Wiener reservoirs with weak continuous measurements and nonlinear static readouts have a rigorous fading-memory and universality result. | Benavoli and Binder, *Phys. Rev. A* **114**, 012433 (2026), [publisher](https://journals.aps.org/pra/abstract/10.1103/ktt8-l2tw). | A positive universality result already exists for a different QRC family. Our conditional density assumption is not a universality proof for the fixed parallel TD-QELM. |
| Reset-and-replay with artificial memory restriction is an established QRC control. | Čindrak et al., *Phys. Rev. Research* **6**, 013051 (2024), [publisher](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.013051). | Truncating replay and observing task changes is not new. |
| Krylov observability quantifies an operator-generated effective phase-space dimension and tracks QRC expressivity or information-processing capacity, including undersampled regimes. | Čindrak, Lüdge, and Jaurigue, *Phys. Rev. Research* **7**, L042039 (2025), [publisher](https://journals.aps.org/prresearch/abstract/10.1103/m9kq-wrln); Čindrak, Jaurigue, and Lüdge, *Phys. Rev. Research* **7**, 043190 (2025), [publisher](https://journals.aps.org/prresearch/abstract/10.1103/84f3-63mz). | Operator-span observability and effective-dimension diagnostics are known. They differ from a covariance-weighted, target-observable conditional-risk floor, but make a generic “observability is new” claim untenable. |
| QELM information access can be characterized through one effective measurement. | Innocenti et al., *Communications Physics* **6**, 118 (2023), [article](https://www.nature.com/articles/s42005-023-01233-w). | General QELM information-retrieval limits are known. |
| A parallel time-delayed QELM resets each row and uploads delayed inputs simultaneously; Appendix E, Eq. (E11), also derives the scalar fixed-linear-readout risk increment from unbiased feature noise with constant covariance. | Kawanabe et al., arXiv:2602.21544v2 (2026), [preprint](https://arxiv.org/abs/2602.21544). | Neither the student architecture nor the core finite-shot quadratic penalty is new. Proposition 2 is a multioutput, input-dependent-covariance restatement used for the matched decomposition. |
| In QELMs, encoding fixes the feature library, the channel mixes it linearly, and measurements determine decodability; PTM rank and measurement budget diagnose access. | Gross and Rieser, arXiv:2602.18377v1 (2026), [preprint](https://arxiv.org/abs/2602.18377). | Measurement/feature accessibility and PTM projection are known for QELMs. |
| Scalable order-statistics expressivity and task-dependent effective rank separate reservoir-family expressivity from information that reaches the readout, with simulated and IBM-hardware validation. | Domingo, Ball{\'o}-Gimbernat, and Vilari{\~n}o, arXiv:2607.09445v1 (2026), [preprint](https://arxiv.org/abs/2607.09445). | A generic expressivity/accessibility distinction is known. Our quantity is a suffix- and prefix-ensemble-specific covariance fraction and conditional risk, not a global expressivity score. |
| Realistic time-series QRC protocols explicitly trade finite-ensemble precision, replay cost, and projective- or weak-measurement back-action. | Mujal et al., *npj Quantum Information* **9**, 16 (2023), [article](https://www.nature.com/articles/s41534-023-00682-z). | Finite measurement resources and measurement back-action in QRC are known. The present frozen-feature shot model describes a narrower reset-and-measure evaluation protocol and does not replace monitored-dynamics theory. |
| A general monitored-QRC framework unifies projective, weak, partial, and dissipative monitoring and derives echo-state, fading-memory, separation, and contractivity criteria with back-action. | Morguí-Sancho et al., arXiv:2607.14311 (2026), [preprint](https://arxiv.org/abs/2607.14311). | Measurement-induced dynamics and online monitoring are distinct from unbiased sampling noise around a fixed ideal feature map; claims must keep those protocols separate. |
| Finite-sampling noise limits supervised learning in physical systems, and correlated quantum features can affect noise-robust capacity. | Hu et al., *Phys. Rev. X* **13**, 041020 (2023), [publisher](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.13.041020). | Sampling-noise covariance and noise-aware feature directions are known; the fixed-readout variance identity is not a novelty claim. |
| Training on finitely sampled QRC and recurrence-free-QRC activations is affected by errors in variables, SVD choices, and denoising. | Ahmed, Tennie, and Magri, *Quantum Machine Intelligence* **7**, 31 (2025), [publisher](https://link.springer.com/article/10.1007/s42484-025-00261-9). | The frozen-readout additive identity does not cover weights trained on noisy features; any noisy-training experiment must be reported as a separate empirical robustness test. |
| Compatible Pauli grouping, full within-group covariance, and covariance-aware shot allocation are established measurement-optimization tools. | Gonthier et al., *Phys. Rev. Research* **4**, 033154 (2022), [publisher](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.033154); Yen, Ganeshram, and Izmaylov, *npj Quantum Information* **9**, 14 (2023), [article](https://www.nature.com/articles/s41534-023-00683-y). | Correlated configuration blocks and square-root-type variance allocation are not new. Their use with a frozen TD-QELM readout is an application, not an invented measurement principle. |
| Delay memory can be formulated locally through a delay-space quantum Fisher information matrix, with finite-shot Pauli readout motivating Fisher-orthogonal modes. | Wang and Qiu, arXiv:2607.29219v2 (2026), [preprint](https://arxiv.org/abs/2607.29219). | This is one of the closest recent measurement-aware memory theories. It concerns local infinitesimal delay estimation, not global old-prefix covariance after a common suffix or the suffix-only Bayes prediction risk. |

### Not new

The following should not be marketed as new mathematics:

- Pauli/Liouville representation of a quantum channel;
- affine cancellation for state differences;
- covariance propagation through a linear map;
- the conditional-mean/Pythagorean risk identity;
- singular-value, Ky Fan, or von Neumann trace inequalities;
- variance propagation through a fixed linear readout;
- compatible Pauli grouping, within-group covariance accounting, and
  square-root allocation for a separable inverse-resource cost;
- QRC fading memory, echo-state conditions, input separation, artificial memory
  restriction, TD-QELM, or QELM feature decodability.

### Candidate contribution after prior-art narrowing

No individual identity above is claimed as new. A bounded audit of the primary
sources leaves the following as a prospective combined contribution:

> To our knowledge, the candidate contribution is the joint use of established
> covariance and risk identities to partition a matched
> QRC-teacher/canonical-parallel-TD-QELM emulation surface into (i) an
> observable-history floor and, after freezing the student readout, (ii) an
> ideal-predictor residual and (iii) a measurement contribution.

The candidate empirical synthesis is therefore **the bridge**, not the
conditional mean, covariance propagation, finite-shot quadratic penalty,
compatible grouping, grouped covariance, or allocation rule:

1. QRC dynamics \(\longrightarrow\) suffix transfer \(T_s\);
2. prefix ensemble \(\longrightarrow\) occupied covariance \(\Sigma_s\);
3. declared teacher observables \(\longrightarrow\) projection \(M\);
4. their product \(\longrightarrow\) exact irreducible emulation risk;
5. frozen canonical TD-QELM features and a specified measurement protocol
   \(\longrightarrow\) exact test-shot penalty;
6. held-out phase map \(\longrightarrow\) observed transitions among
   history-impossible, representation-limited, and measurement-limited regimes.

The result is limited to the declared teacher ensemble, suffix distribution,
teacher observables, student information set and feature map, squared loss,
and measurement protocol. It should be presented as an exact diagnostic
identity plus a prospectively validated phase-map construction, not as a new
general theory of conditional expectation, observability, or quantum
measurement. A bounded literature search cannot establish absolute priority;
a submission should retain “to our knowledge” and attach novelty to the joint
construction and held-out validation package.

### Distinction from the closest adjacent theories

- **Gross--Rieser:** their PTM acts from an encoded QELM feature library to
  measurable features. Here \(T_s\) acts on recurrent teacher-state covariance
  left by old prefixes. Their question is whether a target feature is
  decodable; ours is how much teacher-history variation remains before student
  representation is considered.
- **Čindrak--Jaurigue--Lüdge:** Krylov observability measures an
  operator-generated effective dimension and correlates it with capacity. The
  present quantity weights propagated directions by the actual prefix
  covariance, projects onto declared target observables, and converts the
  result into a conditional squared-risk floor. It does not supersede their
  broader observability diagnostics.
- **Wang--Qiu:** their delay-space QFIM measures local state distinguishability
  of infinitesimal delayed-input perturbations and finite-copy estimation cost.
  Here \(C_s=MT_s\Sigma_sT_s^\top M^\top\) is the classical covariance of a
  finite old-prefix ensemble under a fixed suffix. It yields a global,
  distributional prediction-risk floor rather than a local quantum
  Cramér--Rao bound.
- **Kobayashi--Tran--Nakajima:** their subset/subspace ESP asks whether state
  components become independent of remote initialization or history. Here the
  relevant subspace is selected jointly by prefix covariance and target
  measurement, and the scalar consequence is an exact supervised-risk lower
  bound.
- **Kawanabe et al.:** their Appendix E already gives the scalar finite-shot
  quadratic penalty. Proposition 2 merely records the multioutput,
  input-dependent block-covariance form needed after freezing this paper's
  student; novelty cannot attach to that algebra.
- **Mujal et al. and Morguí-Sancho et al.:** they treat replay, weak or
  projective monitoring, finite resources, and measurement back-action in QRC.
  The present shot term assumes unbiased estimates around an unchanged ideal
  feature map, so it applies only to the stated reset-and-measure protocol and
  does not characterize online monitored dynamics.
- **Hu et al.:** their framework studies resolvable expressive capacity under
  sampling noise. Proposition 2 fixes one affine TD-QELM readout and turns the
  actual compatible-configuration covariance blocks into an exact test-risk
  increment and budget allocation, then adds it to the distinct QRC-history
  floor.
- **Grouped-measurement literature:** full within-group covariance and
  covariance-aware allocation already optimize Pauli-observable estimation.
  Here those tools are reused with readout-induced weights \(W_r\), not proposed
  as a new quantum measurement strategy.

---

## 10. PRA-strength empirical extension

The theorem becomes a substantial physics contribution if the following are
computed without changing the frozen primary decision:

1. **Exact PTM closure test.** Across the registered
   reservoir/system/window/suffix cells, verify direct history MSE against
   \(p^{-1}\operatorname{tr}
   (MT_s\widehat\Sigma_sT_s^\top M^\top)\) to numerical precision.
2. **Hidden-versus-visible memory map.** Plot full post-suffix covariance mass,
   visible covariance mass, and their ratio across \((\gamma_T,d)\). This can
   reveal cells in which the state remembers but the targets do not.
3. **Measurement-family intervention.** At fixed teacher dynamics and prefix
   ensembles, replace \(M\) by X-only, Y-only, Z-only, all single-site,
   same-axis two-site, and full Pauli projections. The theorem predicts every
   history floor without retraining a student.
4. **Spectral mechanism.** Report the singular spectrum of \(MT_sP_s\), the
   covariance spectrum of \(\Sigma_s\), and their alignment. Test whether the
   empirical \(\gamma_T\Delta t d\) collapse comes from contraction of the
   relevant singular modes rather than the full PTM spectral radius.
5. **Observable-active versus full spectral radius.** For constant-symbol and
   representative mixed suffixes, test whether slow full-state modes are
   eliminated by \(M\) or absent from \(\operatorname{ran}(\Sigma_s)\).
6. **Prospective risk prediction.** Use only \(T_s,M,\Sigma_s\) to label
   history-impossible cells before fitting the QELM or classical students.
   After ideal-feature training freezes \((W,b)\), estimate the ideal residual
   on held-out data; only then use \(W_r\) and \(C_r\) to predict the shot term.
   Population identities set the targets, while the held-out estimates require
   uncertainty intervals and are not algebraically identical to them.
7. **Grouped finite-shot validation.** For every compatible experimental
   configuration, estimate the full \(C_r\), compare the observed noisy-readout
   increment with \(\sum_r a_r/S_r\), and prospectively compare uniform
   allocation with the registered square-root allocation at several per-input
   total budgets.
8. **Control ladder.** Include the Bayes conditional mean when computable, a
   flexible classical suffix-only control, the canonical parallel TD-QELM, and
   oracle full-history access. The classical negative result is informative
   in a certified history-impossible cell; success in a cell not excluded by
   the certificate prevents the paper from over-attributing a generic suffix
   bottleneck to quantum architecture.

The focused PRA narrative is:

> Within the tested teacher, prefix ensemble, and observable family,
> observable-accessible covariance determines the suffix-only history floor;
> under the specified frozen TD-QELM sampling protocol, shot budget adds a
> separately quantifiable penalty.

---

## 11. Claim boundary for a paper

### Strongest safe theorem claim

> For any finite-dimensional input-driven QRC, any fixed suffix, any classical
> ensemble of prefix states, and any linear expectation-value readout, the
> observable old-prefix covariance is \(MT_s\Sigma_sT_s^\top M^\top\). Its
> normalized trace is the minimum conditional MSE over square-integrable
> deterministic suffix-only predictors, and hence a lower bound for every such
> predictor, under exact deterministic teacher expectation values and no
> prefix-correlated side information.

### Strongest safe finite-shot claim

> For an affine QELM readout frozen independently of conditionally unbiased
> test-shot noise, finite measurement adds the exact
> readout-weighted covariance penalty
> \(m^{-1}\mathbb E\operatorname{tr}(W^\top\Sigma_\eta W)\). For independent
> configuration blocks, a fixed nonadaptive per-input budget, and equal
> per-shot costs, the unconstrained continuous optimum allocates active
> configurations in proportion to \(\sqrt{a_r}\). This is an application of
> established variance-propagation and resource-allocation identities.

### Strongest safe empirical claim if the extension validates it

> In the tested dissipative spin-QRC and canonical parallel TD-QELM family, the
> emulation surface separates into history-impossible,
> ideal-predictor-residual, and measurement-limited regimes: the population
> history boundary is predicted from \(T_s,M,\Sigma_s\) before fitting, while
> held-out ideal residuals and shot covariances partition the remaining error
> only after the readout is trained and frozen.

### Claims to avoid

- that covariance propagation, conditional variance, or square-root allocation
  is newly discovered mathematics;
- that \(H(s)=0\) means the quantum state has forgotten the prefix;
- that the full PTM spectral radius alone predicts finite-window memory for
  nonnormal or input-varying dynamics;
- that exact expectation access includes finite-shot measurement limits;
- that within-configuration Pauli-feature errors are independent;
- that Proposition 2, compatible grouping, or its shot-allocation rule is new;
- that the finite-shot term is a lower bound beyond the frozen affine readout
  and specified measurement protocol;
- that the theorem establishes quantum advantage;
- that one tested observable family characterizes all possible measurements;
- that the result proves every QRC is approximable by one fixed finite QELM.

---

## 12. Algebra-verification record

The identities and matrix bounds were checked numerically on 28 August 2026
with independent random positive-semidefinite covariances of multiple ranks and
rectangular observable maps:

- direct trace versus Frobenius identity;
- support-restricted minimum/maximum singular-value bounds;
- covariance-eigenvalue/Frobenius bounds;
- von Neumann alignment bounds;
- Monte Carlo convergence to the population history floor;
- exact finite-sample pairwise identity;
- exact fixed-readout finite-shot MSE increment; and
- square-root shot allocation against random feasible allocations.

All checks passed. These numerical checks supplement, but do not replace, the
proofs above.
