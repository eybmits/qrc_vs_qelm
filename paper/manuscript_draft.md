# Task-Dependent Reversal and Observable Memory Fingerprints in Quantum Temporal Learning

> Plain-language evidence synopsis. The current citable manuscript is
> `paper/main.tex` and its compiled `paper/build/main.pdf`.

## One-sentence result

QRC and QELM exchange places as temporal demand increases, while a conservative
output-only recurrence test safely detects finite cutoffs and observable QRC
state carryover but abstains when the frozen measurements expose too little
history signal.

## Confirmed architecture map

- On 15 pre-registered informative Legendre-delay cells: QRC wins 7, QELM wins
  5, and 3 are ties.
- On the full 27-cell grid: QRC 37%, QELM 19%, ties 44%; the unrestricted
  two-regime gate fails because 12 cells are low-capacity for both models.
- QELM beats raw linear delay on the registered nonlinear-cell fraction, showing
  a nontrivial encoded feature map but not a uniquely quantum advantage.

## Held-out recurrence addendum

The pre-registered whole-suite verdict is **NO-GO**. At 10,000 shots, balanced
accuracy is 0.8327 rather than the required 0.85. Every other hard gate and all
integrity checks pass:

- recurrent sensitivity: 128/192 = 0.6667;
- finite specificity and cutoff within one: 767/768 = 0.9987;
- correct decisive labels: 895/960 = 0.9323;
- abstentions: 65/960 = 0.0677;
- observed wrong decisive labels: 0/960;
- false-recurrence calls: 0/768.

Exact expectations give essentially the same balanced accuracy. All eight
dephasing QRC blocks fall below the frozen accessible-signal floor, while all 16
local/collective-loss blocks are resolved. This is a post-hoc visibility
partition, not proof that dephasing destroys memory.

The separately registered task test passes: the QRC-minus-QELM ordering shifts
by 0.5364 from short to long delays, with a 95% physics-cluster interval
[0.4333, 0.6448] and 23/24 positive system shifts.

## Claim boundary

The evidence supports a simulator-bounded task reversal, an abstention-safe
recurrence proof of concept, and an observable-memory accessibility boundary.
It does not support universal QRC superiority, arbitrary finite-QELM failure,
quantum advantage, hardware performance, physical non-Markovianity, or the
claim that dephasing erases all stored information.
