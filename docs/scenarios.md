# Scenarios & state definitions

> Draft. Finalize with Ömer on day 1-2, then keep in sync with
> `configs/pipeline.yaml` (`expected:` block).

## Objects

Three distinguishable items of different sizes, can overlap. Named functionally
in code: `item_a`, `item_b`, `item_c`. (Physical mapping kept out of the repo.)

| Class | Rough size | Notes |
|-------|-----------|-------|
| item_a | large | |
| item_b | medium | |
| item_c | small | most occlusion-prone |

## Expected configuration

- All three present.
- Arranged in a single row, left → right: `item_a, item_b, item_c`
  (ordered by bounding-box centroid x).

## Verdicts

| Verdict | Trigger |
|---------|---------|
| `OK` | all three present, correct order |
| `MISSING` | ≥1 expected item absent, no unexpected items |
| `WRONG_ORDER` | correct set of items, wrong left-to-right order |
| `WRONG_COMBINATION` | an unexpected / duplicate item present |
| `EMPTY` | nothing detected |

## Test scenarios (for days 18-20)

1. Baseline OK.
2. Remove item_c → `MISSING`.
3. Remove item_a and item_b → `MISSING`.
4. Swap item_a and item_b → `WRONG_ORDER`.
5. Reverse all three → `WRONG_ORDER`.
6. Add a duplicate item_b → `WRONG_COMBINATION`.
7. Partially occlude item_c by hand for < smoothing window → stays `OK`
   (smoothing rejects the blip).
8. Occlude item_c for > window → `MISSING`, then restore → back to `OK`.
9. Lighting change (lamp on/off) → verdict unchanged.
10. Fast hand pass across the scene → no spurious transition.

Record for each: expected verdict, observed verdict, frames-to-detect,
any false alarm.
