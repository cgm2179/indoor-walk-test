# Assessment: tensor-network methods for this pipeline

You asked whether TT-Cross / MALS / AMEN / randomized-TT / QTT / MaxVol would
help — for dataset generation speed, or in the ML architecture. Honest answer:
**they don't fit this problem, and adding them would be complexity without
payoff.** Reasoning, per method and per stage.

## Where the time actually goes (the thing to speed up)
Dataset generation = for each of 2,500 transmitters, trace 9 bands of ray
geometry over a 256×448 grid on the GPU. The cost is **dense, structured ray
tracing**, already vectorized in torch (numpy↔torch parity 0.0007 dB) and LUT-
accelerated (geometry traced once, all 9 bands via two gathers + a lerp). On a
T4/A100 the full run is minutes. There is no high-dimensional black-box function
being sampled — it's a fixed, cheap, fully-parallel kernel.

## Why each tensor method is a mismatch here
- **TT-Cross / TT-CI (active learning):** shines when a function of many
  variables is *expensive* to evaluate and you want to avoid evaluating the
  full grid. Our "function" (path loss) is *cheap* per point and we *want* every
  cell of every map (the target IS the dense map). There's nothing to skip —
  TCI's whole value (fewer evaluations via pivots) buys nothing when you need
  the full tensor as output.
- **MALS / AMEN (adaptive-rank cross):** same category; they reduce black-box
  calls for high-rank functions. Our bottleneck is throughput of a known kernel,
  not call count.
- **Randomized SVD / randomized TT:** fast *low-rank compression* of an existing
  matrix/tensor. A single path-loss map is a 256×448 image with sharp wall
  discontinuities — **not** low rank (that's exactly why we train a U-Net rather
  than an SVD). Compressing it would throw away the wall-shadow structure the
  surrogate must learn.
- **Quantics TT (QTT):** turns O(n) into O(log n) for functions on *enormous*
  1-D/3-D grids (2^20+ points/axis) that are smooth/low-entanglement. Our grid
  is 256×448 (~10^5 cells) and the field is discontinuous at walls — neither the
  scale nor the smoothness QTT needs. O(log n) of a small n saves nothing.
- **MaxVol (pivot selection):** only meaningful inside TCI/cross, which we're not
  using.

## The one place a related idea *could* help (and why we already do it)
The genuinely useful compression insight — "evaluate strategic pivots, not the
whole space" — is already realized in two forms:
1. **The CrossingLUT:** the (material, cos θ, thickness) → loss surface is
   smooth and low-dimensional, so we tabulate it once (~10 ms) and interpolate,
   instead of solving the Fresnel slab per crossing per band. That IS the
   "sample strategic points of a smooth EM sub-function" idea, applied where it
   fits.
2. **The relay cache:** the obstruction map radiating from a diffracting corner
   is transmitter-independent, so we precompute a spread of corners once per
   grid rather than per transmitter. Same principle, different axis.

## The ML architecture
The surrogate is a 2-D U-Net (map → map). It already exploits spatial structure
via convolutions. A TT/tensor-network layer would help only for *very high-order*
tensor inputs (e.g., a 5-D+ parameter cube), which this isn't — the input is a
10-channel image. No benefit; it would slow training and complicate export.

## Bottom line
- **Dataset generation:** already GPU-parallel and LUT-accelerated; the tensor
  methods target a problem shape (expensive high-D black box, low-rank output)
  we don't have. No change recommended.
- **If generation ever becomes the bottleneck**, the effective levers are:
  batch multiple transmitters per kernel, fp16 tracing, and multi-GPU sharding
  (`--shard-mod/--shard-rem`, already supported) — not tensor cross methods.
- **ML side:** convolutional U-Net is the right inductive bias for map→map; a
  tensor-network layer is a mismatch for image-shaped I/O.

Kept here so the reasoning is on record if the question comes up again.
