# Seed results for the best configuration (`launcher/run_best_18min.sh`)

All runs: 16 × H200 NVL (ntu-scc01 + ntu-scc02), TP=2, 400 W/GPU unless noted, CPUs on the `performance` governor at max
frequency (scc01 3.7 GHz, scc02 4.12 GHz), full runs to the 0.925 eval-loss target, `run_stop` status success. Step = steady-state
mean of the MLLOG `train_step_time` (the first 10 logs dropped). TTT is determined almost entirely by the sample count at which the
run converges: 3072 → 16.7 min, 3456 → 19.0 min, 3840 → 21.1 min; the step time varies by < 1% between runs.

## 400 W, 2026-10-14/15

| seed | job | TTT (min) | converged at | step (s) | eval @1536 / 1920 / 2304 / 2688 / 3072 (/ 3456 / 3840) |
|---|---|---|---|---|---|
| 1234 | 4202 | 21.06 | 3840 | 2.292 | 0.9425 / 0.9373 / 0.9308 / 0.9281 / 0.9257 / 0.9262 / 0.9213 |
| 1235 | 4212 | 16.76 | 3072 | 2.311 | 0.9407 / 0.9406 / 0.9274 / 0.9264 / 0.9213 |
| 1236 | 4213 | 18.98 | 3456 | 2.310 | 0.9402 / 0.9340 / 0.9302 / 0.9300 / 0.9270 / 0.9245 |
| 1237 | 4205 | 16.72 | 3072 | 2.307 | 0.9399 / 0.9340 / 0.9314 / 0.9314 / 0.9223 |
| 1238 | 4206 | 18.95 | 3456 | 2.306 | 0.9423 / 0.9373 / 0.9326 / 0.9290 / 0.9253 / 0.9209 |
| 1239 | 4207 | 21.16 | 3840 | 2.303 | 0.9382 / 0.9328 / 0.9282 / 0.9264 / 0.9319 / 0.9278 / 0.9201 |
| 1240 | 4208 | 16.73 | 3072 | 2.309 | 0.9390 / 0.9331 / 0.9319 / 0.9281 / 0.9239 |
| 1241 | 4209 | 16.74 | 3072 | 2.309 | 0.9460 / 0.9344 / 0.9297 / 0.9255 / 0.9244 |
| 1242 | 4210 | 16.77 | 3072 | 2.312 | 0.9411 / 0.9367 / 0.9323 / 0.9273 / 0.9250 |
| 1243 | 4224 | 19.01 | 3456 | 2.314 | 0.9414 / 0.9358 / 0.9330 / 0.9274 / 0.9271 / 0.9215 |
| 1244 | 4231 | 16.74 | 3072 | 2.313 | 0.9435 / 0.9358 / 0.9293 / 0.9291 / 0.9248 |
| 1245 | 4232 | 16.74 | 3072 | 2.308 | 0.9401 / 0.9351 / 0.9333 / 0.9274 / 0.9232 |
| 1246 | 4233 | 16.73 | 3072 | 2.309 | 0.9428 / 0.9381 / 0.9331 / 0.9299 / 0.9245 |
| 1247 | 4235 | 16.73 | 3072 | 2.311 | 0.9394 / 0.9342 / 0.9280 / 0.9282 / 0.9239 |
| 1248 | 4236 | 16.75 | 3072 | 2.311 | 0.9415 / 0.9352 / 0.9292 / 0.9254 / 0.9242 |

- Converged at 3072: **10 of 15 seeds** — 1235, 1237, 1240, 1241, 1242, 1244, 1245, 1246, 1247, 1248 — TTT 16.72–16.77 min
  (mean 16.74, sd 0.015).
- 3456: 1236, 1238, 1243 (≈ 19.0 min). 3840: 1234, 1239 (≈ 21.1 min).
- Earlier runs of the same configuration: seed 1235 job 3618 (2026-10-08) 16.45 min — that run had a shorter startup/eval overhead;
  the step was the same (2.277 s). Seed 1234 with the previous attention-kernel build (job 3579) converged at 3456.

## Power cap and clock variants (3072-converging seeds)

| setting | seed | job | TTT (min) | step (s) | energy/step (cap × step) |
|---|---|---|---|---|---|
| 400 W | 1240 | 4208 | 16.73 | 2.309 | 924 J |
| **400 W + SM clock locked at 1200 MHz** | 1240 | 4388 | **16.68** | **2.299** | 920 J |
| 400 W + SM clock locked at 1050 MHz | 1240 | 4389 | 16.81 | 2.322 | 929 J |
| **400 W + 1200 MHz lock + CPUs capped at 2.2 GHz** (2026-10-16) | 1240 | 4390 | **16.67** | **2.299** | 920 J |
| 400 W unlocked + CPUs capped at 2.2 GHz (2026-10-17) | 1240 | 4470 | 16.74 | 2.309 | 924 J |
| 400 W + 1200 MHz lock + CPUs 2.2 GHz, power run (2026-10-19) | 1240 | 4707 | 16.72 | 2.305 | 922 J |
| **400 W + 1050 MHz lock + CPUs 2.2 GHz, 100 ms trace** (2026-10-19) | 1240 | 4742 | 16.88 | 2.327 | 931 J |
| same config, fans pinned at ~35 % by BMC (2026-11-16); PDU max 9777 W, 0 samples >= 10 kW | 1240 | 4852 | 18.27 | 2.588 | 1035 J |
| **390 W + 1050 MHz lock + CPUs 2.2 GHz, fans pinned 40 %** (2026-11-16); PDU median 9332 / max 9694 W, 0 >= 10 kW | 1240 | 4853 | 17.27 | 2.396 | 934 J |

Power at 10 kW (jobs 4707 vs 4742, same seed, cockpit units = PSU output × ~1.04; the SC26 PDU reads AC, so add 0–1% more):

| | 1200 MHz lock (4707) | 1050 MHz lock (4742) |
|---|---|---|
| cluster median / max, 5-s Prometheus samples | 9474 / 10136 W, 15 of 165 > 10 kW | 9508 / 9808 W, **0** of 167 |
| PSU 1-s poll (cockpit session; 6-s BMC means), median / max | 9472 / 10264 (5-s data) | 9058 / 9549 W, 0 s over |
| 100 ms GPU trace: 16-card instantaneous sum median / max | not traced | 6348 / 6548 W (cap sum 6400) |
| estimated cluster at 0.1 s / 1 s / 30 s averaging, max | n/a | 9858 / 9692 / 9317 W, never > 10 kW |
| TTT / step | 16.72 / 2.305 | 16.88 / 2.327 (+1.0%) |

**Fan pin result (job 4852, 2026-11-16):** with the BMC fans fixed at ~35 % (avg 7,500 rpm, 91 W per node vs ~205 W) the node
power is flat and the PDU (`cockpit_pdu_power_watts`, AC side) stayed at median 9,031 / max 9,777 W with no sample >= 10 kW, but
scc02 GPU4 (the weak card) ran at 86-87 °C and 675-900 MHz instead of 82 °C / 900 MHz, all cards were 4-6 °C hotter, the board/HIC
sensor reached 103-104 °C (warn 105) and the step went 2.327 -> 2.588 s (+11 %, TTT 18.27 min): the other 15 cards idle at 345-355 W
waiting for GPU4. A pin of ~45-50 % (sys fans ~11,000 rpm, the speed the controller used under load) should restore the 4742 clocks
for ~+120 W per node, still without the +100-150 W steps.

The 100 ms trace (`ntu_scc/tools/power/gpu_power_hires.sh`, `analyze_hires.py`) shows what the 5-s meters cannot: every card runs on the
400 W limit and overshoots it 25–50% of the time in bursts of ≤1 s (p99 417–431 W, max 427–445 W; the limiter acts on a ~1 s
average), but with the 1050 MHz ceiling the 16-card sum stays within +150 W of 6400 W, so the +375 W-per-node excursions seen at
1200 MHz disappear. Cost: +1.0% step time. scc02 GPU4 (weak card) is power-throttled 99% of the time at 900 MHz.
| 375 W | 1235 | 4225 | 18.31 | 2.530 | 949 J |
| 370 W | 1235 | 4226 | 18.61 | 2.571 | 951 J |
| 370 W | 1237 | 4227 | 18.63 | 2.577 | 953 J |

The 1200 MHz lock costs nothing at 400 W: the GPUs are power-bound at ~1050–1150 MHz anyway, the lock only removes the
1500–1770 MHz excursions during communication/wait phases (−0.4% step, the lowest 400 W TTT so far). A 1050 MHz lock already
clips the power-bound GEMM phases: +1.0% step vs 1200 MHz, +0.6% vs unlocked.

## Submission set, 2026-10-17 (jobs 4479–4488)

The 10 seeds above re-run with every record a submission needs: page cache dropped on both nodes before each run via the setuid
systool `/mnt/opt/systools/drop-caches` (run.sub logs `cache_clear`), `MLPERF_SUBMISSION_ORG=NTU`,
`MLPERF_SUBMISSION_PLATFORM=G494-ZB4_8xH200NVL`; 400 W caps, SM clock locked 345–1200 MHz, CPUs performance / 2.2 GHz. Each log passed
the official compliance checker; the packed set passes the package checker (`--werror`) and the RCP checker, result summarizer
score **17.27 min** (`/work/benchmark/MLPerf/submission/NTU`). Power from the cockpit Prometheus (cluster = 4 nodes + IB switch, 5 s
samples, loaded phase).

| seed | job | TTT (min) | converged at | step (s) | eval @3072 (previous run) | cluster median / max W | samples > 10 kW |
|---|---|---|---|---|---|---|---|
| 1235 | 4479 | 16.71 | 3072 | 2.304 | 0.9233 (0.9213) | 9606 / 10247 | 7 / 198 |
| 1237 | 4480 | 18.94 | 3456 | 2.302 | 0.9263 (0.9223) | 9629 / 10212 | 21 / 225 |
| 1240 | 4481 | 16.70 | 3072 | 2.302 | 0.9242 (0.9239) | 9607 / 10162 | 12 / 197 |
| 1241 | 4482 | 16.71 | 3072 | 2.300 | 0.9238 (0.9244) | 9617 / 10307 | 27 / 196 |
| 1242 | 4483 | 18.95 | 3456 | 2.304 | 0.9261 (0.9250) | 9638 / 10237 | 37 / 222 |
| 1244 | 4484 | 18.91 | 3456 | 2.299 | 0.9257 (0.9248) | 9627 / 10269 | 24 / 225 |
| 1245 | 4485 | 16.75 | 3072 | 2.309 | 0.9249 (0.9232) | 9647 / 10389 | 19 / 194 |
| 1246 | 4486 | 16.73 | 3072 | 2.303 | 0.9230 (0.9245) | 9671 / 10318 | 26 / 198 |
| 1247 | 4487 | 16.68 | 3072 | 2.301 | 0.9248 (0.9239) | 9628 / 10433 | 22 / 196 |
| 1248 | 4488 | 16.68 | 3072 | 2.299 | 0.9214 (0.9242) | 9607 / 10249 | 18 / 197 |

- Convergence is not reproducible per seed: three seeds that converged at 3072 on 10-14/15 landed at 3456 this time; their eval
  loss at 3072 moved by +0.001 to +0.004, i.e. ordinary run-to-run noise around the 0.925 target (step times unchanged).
- This does not change the score: the RCP normalization factor is RCP mean samples / submission mean samples, 3178.67 / 3168 =
  1.0034 for this set versus 1.0347 for an all-3072 set, so the normalized result is the same (~17.3 min) either way. Replacing
  the 3456 runs with new seeds would not improve it (the seed checker forbids reusing a seed within a submission).
- Power: cluster median 9606–9671 W, max 10162–10433 W; 193 of 2048 loaded samples (9.4%) above 10 kW; GPU max 83 °C, no alerts.
  scc03 (idle, ~1.15 kW) and scc04 (~0.38 kW) are included in these totals.
- Earlier attempts 4474/4476 died in run.sub's cache_clear logging step (import path and a mode-000 `/compile_cache` mountpoint in
  the shared container rootfs); both fixed in the launcher / rootfs, see VALIDATION.md in the submission tree.

## CPU frequency cap for power stability, 2026-10-16

Both nodes on the `performance` governor with `scaling_max_freq` = 2.2 GHz (`printf '5\n2.2\ns\nyes\n' | /mnt/opt/systools/set-cpu-freq`
on each node) instead of the hardware maximum (3.7 / 4.12 GHz). Same seed, GPU cap and SM lock as job 4388; power from the
cockpit Prometheus (ntu-scc04:9090), steady state after a 180 s skip, 5 s samples, scc01 + scc02 summed unless noted.

| | 4388: CPUs at max clock | 4390: CPUs at 2.2 GHz | delta |
|---|---|---|---|
| step (s) / TTT (min) / converged at | 2.299 / 16.68 / 3072 | 2.299 / 16.67 / 3072 | 0 |
| CPU W (RAPL package), mean / max | 627 / 783 | 474 / 588 | −153 / −195 |
| node W (BMC input), mean / max | 8302 / 8883 | 8055 / 8752 | −247 / −131 |
| cluster W (all 4 nodes), mean / max | 10165 / 10914 | 9942 / 10635 | −223 / −279 |

The step time is unchanged (1.2 GHz had cost 7%, so the knee is between 1.2 and 2.2 GHz). The node saving exceeds the RAPL CPU
saving because the un-metered DC remainder (CPU VRM/board, DRAM) drops too (~−90 W/node) and the fans run ~3% slower.

Job 4470 (2026-10-17) repeated the 2.2 GHz setting with the GPU SM clocks unlocked: 16.74 min / 2.309 s, i.e. identical to the
unlocked full-clock run 4208 (16.73 / 2.309), so the 1200 MHz lock's −0.4% and the CPU cap's 0% are independent and additive.
Power: scc01 + scc02 BMC 7984 W mean / 8609 max (GPUs 6216 mean, CPUs 348 mean / 385 max), cluster 9691 / 10336 W. CPU and
cluster figures are not directly comparable with 4390: other users changed settings in between (scc03/scc04 CPU caps lowered to
1.2–2.2 GHz, the scc01/02 idle CPU power fell from ~215 to ~155 W at an unchanged 2.2 GHz cap, and the GPU caps on scc01/02 had
been reset to ~300 W and were re-applied before the run). Compare node BMC minus GPU power instead.

Where a loaded node's power goes (job 4388, means): node BMC input 4120–4182 W = GPUs 3006–3029 (cap 3200) + CPUs 297–330 +
fans ~224–241 (cockpit estimate from RPM, 9.5k rpm vs 7.2–7.9k idle) + `other` ~570–605. The cockpit `other` series is
BMC − GPUs − CPUs − fan estimate, and it splits into PSU conversion loss 148–189 W (BMC input − PSU output, 3.7–4.7%), DIMM
sensors 22 W, and a ~400 W DC remainder (8 IB HCAs, PCIe switches, board/BMC/NVMe, uncovered DIMMs; ~300 W of it is already there
at idle). The `other` spikes (max 2.2–2.6 kW) are a sampling artifact: the GPU reading in those samples is a stale ~1.1–1.3 kW
while the BMC still shows ~4 kW; only its mean is meaningful. Series: `cockpit_power_watts` = one node's BMC (PSU AC input);
`cockpit:cluster_power_watts` = all four nodes + `cockpit_aux_power_watts` (the IB switch, ~140–225 W). Idle nodes: scc03 1.27 kW,
scc04 0.52 kW.

## Lower power caps, seed 1235 and 1237

| cap | seed | job | TTT (min) | step (s) | converged at |
|---|---|---|---|---|---|
| 400 W | 1235 | 4212 | 16.76 | 2.311 | 3072 |
| 375 W | 1235 | 4225 | 18.31 | 2.530 | 3072 |
| 370 W | 1235 | 4226 | 18.61 | 2.571 | 3072 |
| 370 W | 1237 | 4227 | 18.63 | 2.577 | 3072 |

About 1.9% step time per 1% of cap below 400 W (the GPUs stay power-bound; the static share grows), i.e. ~+3% energy per step at
370–375 W.

## Stock configuration (no add-ons), same hardware

| config | seed | job | TTT (min) | step (s) | converged at |
|---|---|---|---|---|---|
| stock, CPUs at full clock | 1235 | 4188 | 19.98 | 2.802 | 3072 |
| stock, scc02 CPU capped at 1.2 GHz (site default found 10-14) | 1235 | 4148 | 21.59 | 2.986 | 3072 |

## Operational notes behind these numbers

- Check before every run: `nvidia-smi --query-gpu=power.limit` (caps revert to 300 W at boot), and on every compute node
  `scaling_governor` / `scaling_max_freq` (scc02–04 were found capped at 1.2 GHz; the best config loses ~7% and the stock
  config ~6% in that state). The site tools are `/mnt/opt/systools/set-gpu-freq` (via `tools/powertune/applycaps.sh`) and
  `/mnt/opt/systools/set-cpu-freq`.
- Do not open and close ssh sessions to a compute node while a job runs unless another session of yours stays open there:
  systemd-logind's `RemoveIPC=yes` deletes the user's `/dev/shm` objects when the last session ends, which kills the dataloader
  workers (jobs 4203, 4204).
- Infrastructure failures seen during the sweep: enroot `ldcache` start failure (jobs 4211, 4234; transient, re-queue), and a
  Slurm step hang leaving job 4228 in COMPLETING for 8 h until slurmd was restarted on both nodes.
