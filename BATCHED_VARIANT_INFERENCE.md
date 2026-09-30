# Batched variant inference — design note

Status: **not implemented**, deliberately deferred. Written 2026-08-26.

## The idea in one sentence

When several *variants of the same event* are reconstructed by *the same
weights*, they are independent graphs and can go through the model as a single
batch instead of N separate forward passes.

Two things produce such variants:

- **Pulse-exclusion arms** (`Steps/ablation.py` in Selection-Processing): the
  same event under different `exclude_flags`. N = 4.
- **Perturbation ensembles**: the same event with `perturbation_dict` sampled
  N times. N = 50, 100, whatever you want.

Only the second one really needs this. See "When it is worth building".

## Where things stand today

`I3InferenceModule` has a `multiple_models=True` mode that already does the
*hard* part: it calls the I3 extractor **once** per frame and builds one data
representation per model
(`deployment/icecube/inference_module.py:369`, `:209`).

What it does not do is batch. Per frame it builds N single-graph batches and
runs N forward passes:

```python
# inference_module.py:237
model_input_data.append(Batch.from_data_list([data.to(self._device)]))
...
# deployment_module.py:151
for _, model in enumerate(self.models):
    output = model(data=data[_])
```

and it holds N *separate* `Model` instances (`_load_model`,
`deployment_module.py:74`, called once per (config, state_dict) pair), so
there is currently nothing to batch *into* even if you wanted to.

The output side is already N-aware: `_create_dictionary`
(`inference_module.py:288`) walks per-model column offsets, and
`I3ParticleInferenceModule._add_to_frame` (`:556`) already loops
`self._model_names` building one I3Particle per variant. That work was done
when `multiple_models` was made multi-task capable, and it is why this change
is small.

## Proposed change

Add a mode where the variants share one model. Not a rewrite — roughly 120
lines across methods that already exist.

| method | file:line | change |
|---|---|---|
| `_load_model` | `deployment_module.py:74` | load once when the state_dicts are identical; keep N data representations |
| `_resolve_prediction_columns` | `deployment_module.py:89` | one shared label list rather than one per model |
| `_inference` | `deployment_module.py:132` | **core.** one `model(data=batch)`; each task output is `(N, k)`; split rows by variant |
| `_create_data_and_apply` | `inference_module.py:209` | **core.** collect N `Data` and `Batch.from_data_list(datas)` once |
| `_check_dimensions` | `inference_module.py:254` | expect `(N, k)` instead of a flat concatenation |
| `_create_dictionary` | `inference_module.py:288` | map row *i* to variant *i*'s key prefix (already loops names) |
| `_add_to_frame` | `inference_module.py:556` | **no change needed** |

Variable-size graphs are not a problem — collating graphs of different node
counts is exactly what `Batch.from_data_list` is for, and the model was
trained on batches, so batched inference is its native path.

Keep it behind a separate flag rather than changing what `multiple_models`
means. `multiple_models` legitimately means "N different models" and is used
that way by the bundle classifiers in Selection-Processing `Steps/step_2.py`,
which must keep working byte-identically.

## Two traps

**1. `_perturb_input` mutates its argument in place.**

```python
# models/data_representation/data_representation.py:315
input_features[:, self._perturbation_cols] = perturbed_features
```

Today each variant receives its own array from
`_extract_feature_array_from_frame`, so this is safe. The obvious
optimisation — extract once, hand the same array to all N variants — would
make perturbations **compound**: variant 3 would be perturbed three times.
Any implementation must copy per variant, and that copy is not free for a
40k-pulse event.

**2. Seeding is per-instance.**

`self.rng` lives on the `DataRepresentation` instance
(`data_representation.py:91-99`), and `seed` is a constructor argument. N
perturbed variants therefore need N distinct seeds baked into N configs, or a
way to reseed after construction. Decide which before building, because it
determines whether the ensemble is reproducible.

## Measurements that led to deferring this (2026-08-26)

Hardware: 3x RTX 6000 Ada (48 GB) + 1x H100 NVL (96 GB), 64x Xeon Gold 6448H.
Workload: 4-arm ablation, `lowe_v4` checkpoint, one nugen file (49 events,
13k pulses/event).

| config | repr s/ev | infer s/ev | total s/ev |
|---|---|---|---|
| CPU, 2 threads | 0.037 | 9.562 | 9.600 |
| CPU, 8 threads | 0.037 | 9.434 | 9.472 |
| RTX 6000 Ada | 0.036 | 0.237 | 0.273 |
| H100 NVL | 0.035 | 0.275 | 0.310 |

- **GPU is ~35x faster.** CPU thread count is not a lever (8 threads bought
  1.3% over 2) — at batch size 1 the transformer does not parallelise across
  cores.
- **The two GPUs tie**, with the cheaper card marginally ahead. That is a
  latency-bound signature and is the main *argument for* batching.
- **But four processes sharing one card each ran at ~0.16 s/event** — faster
  than one process alone — at 30% utilisation and ~750 MiB per process. You
  can fill the GPU with processes, which is free and needs no code.

For a throughput job, more processes is the cheaper lever than lower latency.
At N=4 the optimisation was not worth the risk.

## When it is worth building

When N stops being 4.

A perturbation ensemble with N=50-100 replicas per event pays the per-call
overhead 50-100 times per event, and process-level parallelism cannot amortise
it away — every replica is still its own forward pass. There, batching is the
difference between feasible and not, and the trap about in-place perturbation
becomes load-bearing rather than academic.

Trigger to revisit: any use case with N >= ~16 variants per event, or a
latency-sensitive (rather than throughput-sensitive) deployment.
