# Skill: all-14 four-surrogate five-seed campaign v1

## Scientific contract

Run 280 fresh arms: 14 independently configured disordered systems × four online surrogate agents × five independent seeds (0--4). The four agents are ExtraTrees, Bayesian ridge, sparse ARD, and the compact fixed-topology ALIEGNN-inspired energy ensemble. NaSbSe2 is not in this campaign.

Every arm starts from a fresh seed-local initial occupancy, optimizer, history, elite archive, surrogate/model state, output directory, and calculator instance. The same system and seed must produce the same initial occupancy across all four agents. No completed arm may be reused from an earlier campaign.

## Systems

`AlReGe`, `Ba2ScTaO6`, `CoPt`, `Cu3Au`, `Cu3Au_123`, `FeCo`, `FeCo_331`, `FePt`, `InCuTe2`, `LiYS2`, `Ni2SbTe2`, `ThMn2Co3`, `TiCuNi`, and `V2ZnO4`.

The optimizer must not hard-code any formula, site count, element identity, framework composition, or active-species count. For each system, derive the active sites, exact two-species integer composition, fixed framework, supercell, geometry, descriptor, and graph from that system's captured current config/CIF. Reject a non-binary active occupancy or unsupported resource size before calculator setup.

## Objective and accounting

Minimize TensorNet-MatPES-PBE-2025.2 relaxed total energy at exact fixed composition. Step 0 is the initial structure. Steps 1--200 are proposals; proposal 201 is forbidden. Two structures are comparable only within the same system and calculator/relaxation contract.

## Runtime isolation

During selection, an arm may access only:

- this frozen Skill and captured source buffers;
- captured current config and CIF bytes for its own system;
- observations generated in that arm.

Runtime access to enumeration/reference databases, earlier trajectories, other systems or arms, known structures, motifs, fitted models, or precomputed energies is forbidden. Post-run reference comparison is permitted only after all 280 arms have finalized and passed integrity checks.

## Shared search policy

All agents share the same:

- current-input CE-inspired point/pair/triplet descriptor for novelty;
- maximum-four diverse elite archive with minimum Hamming distance four;
- unseen unlike-species single exchanges;
- balanced 2-, 3-, and 4-pair exchanges, lazily unranked and capped at 3000 candidates per acquisition;
- novelty warm-up until eight observations including step 0;
- requested greedy, uncertainty, hybrid LCB-plus-novelty, and novelty acquisitions;
- coordinated-jump schedule and coverage accounting;
- TensorNet evaluator and exact composition checks;
- evidence-based stopping criteria and hard proposal ceiling.

The hybrid LCB score is mean minus 1.5 times uncertainty minus 0.08 times current energy scale times normalized novelty. Requested and actually executed acquisition modes must both be recorded.

## Surrogate agents

1. **ExtraTrees:** 400 trees, minimum leaf size 2, max-features fraction 0.9, four workers; mean and tree-disagreement standard deviation.
2. **Bayesian ridge:** standardized CE-inspired descriptor, isotropic Gaussian shrinkage, posterior predictive mean and standard deviation.
3. **Sparse ARD:** same descriptor and acquisition semantics, feature-wise precisions, pruning threshold 10,000, posterior predictive mean and standard deviation.
4. **ALIEGNN-inspired:** no pretrained checkpoint or external training data; three independently initialized/bootstrap compact atom/angular-line-graph models, width 16, one block, 20 full-batch AdamW epochs per fit. Raw ensemble spread plus current-fit residual floor is an uncalibrated uncertainty proxy. Store ordered training-data and fitted-state digests plus prequential residuals.

## Fair execution order

Within every system and seed, rotate the four surrogate order by `(system_index + seed) mod 4`. Record the exact order. This counterbalances warm-up, temporal load, and thermal effects across systems and seeds while keeping TensorNet arms strictly serial.

## Provenance and failure safety

- Launch the runner through the single-read captured-byte executor.
- Execute all local helper/policy modules from their captured buffers.
- Hash and archive Skill, runner, policies, utilities, config, CIF, resolved structure, graph, package versions, and actual TensorNet state.
- Guard mutable paths and TensorNet state around every expensive evaluation and immediately before final publication.
- Claim the campaign lock and every arm directory atomically.
- Publish status and final summaries atomically.
- Exit nonzero immediately on the first failed arm; do not spend later arms after the exact-complete campaign has already become invalid.
- Retain a failed campaign lock for deliberate inspection.

## Resource limits

Reject jump spaces above the supported exact finite bound before calculator setup. ALIEGNN candidate batches are capped at 3000, history at 201, graph nodes at 256, neighbor parameter at 64, and prospective line-graph edges at one million. ExtraTrees uses four workers. Launch only through `campaign/launch_all14_four_surrogates_five_seeds_v1.sh`, which creates a dedicated user systemd unit with `MemoryHigh=16G`, `MemoryMax=24G`, `MemorySwapMax=4G`, `TasksMax=128`, `CPUQuota=1600%`, `OOMPolicy=kill`, and `RuntimeMaxSec=7d`. These limits retain headroom beyond the observed TensorNet-plus-surrogate task count while bounding the multi-day campaign independently from the chat gateway.

## Required preflight

Before launch:

1. Resolve all 14 configs from captured current inputs.
2. Verify binary active species, exact integer counts, composition preservation, graph/resource bounds, and descriptor dimensions.
3. Verify all 280 output directories are absent and the campaign lock is absent.
4. Run focused unit and contract tests.
5. Run mocked end-to-end arms across all four selection branches and across representative site counts/compositions.
6. Obtain an independent read-only GO audit bound to exact hashes.
7. Verify GPU/process state and launch exactly one serial campaign.

## Reporting

For every system, plot five traces per agent, bold mean, and ±1σ shading on proposals 0--200. Produce per-system figures, a readable contact sheet, per-run and aggregate CSV files, proposal-to-best, wall-clock-to-best, final energy, fit overhead, and reference-hit/rank panels. Distinguish this four-agent campaign from random/MC/GA baselines and from live per-step LLM inference.
