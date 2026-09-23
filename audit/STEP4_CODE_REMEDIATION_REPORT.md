# Step 4 Code-Remediation Gate Report

Branch: `fix/scientific-pipeline-v2`  
Starting commit: `a65290a7c52151c5904ac9accb1798fd6dc788f5`  
Starting tree: `7b99aaa3f09ddbd9ee1250034a93d057b7d157ba`

## Gate outcome

PASS — code remediation may proceed to new FINAL_V2 experiments.

- Python regression command: `python -m unittest discover -s tests -v`
- Tests: 11 passed, 0 failed.
- Dependency command: `python -m pip check`
- Dependency result: `No broken requirements found.`
- Source syntax: 43 Python source/script files parsed.
- Dataset snapshot verification: PASS, manifest SHA-256
  `a94426015d83a3b8e9f66a30bddee9e4b376eaf509f166aa522d7a204fe22dc5`.
- Frozen configuration invariants: 4/4 backbone configs passed.

The environment emitted a warning about an invalid `-illow` distribution entry while
`pip check` still reported no broken requirements. Step 4 did not install, remove, or
repair packages.

## Confirmed remediation scope

- Fixed external validation now controls OOF checkpoint/model selection.
- Outer folds are inference-only after checkpoint freeze.
- Weighted Voting uses SLSQP and multiclass log-loss.
- Stacking configuration is centralized and fitted objects are serialized.
- Verification replays prediction/model artifacts without refitting.
- Protocol roots, samples, labels, class order, and shapes are validated.
- Scientific fallbacks fail closed.
- Validation and test artifacts have explicit names.
- Statistical decisions use raw exact p-values and neutral decisions.
- t-SNE source/dimension metadata is explicit.
- Latency code measures saved pipelines and actual files/parameters.
- New runs record dataset/source/config/runtime/artifact provenance.
- Historical results are not overwritten; new outputs target `RESULTS/FINAL_V2`.

## Known unresolved items at this gate

- FINAL_V2 numerical results do not exist until Phase S completes.
- Real latency values, verification tables, and final consistency hashes cannot be
  produced before those new checkpoints/predictions exist.
- The pre-existing deletion of `docs/codebase_documentation.md` remains untouched.
- The Step-1/Step-2 audit files referenced by the request were absent at Step-4 start;
  Step 4 did not reconstruct them.

## Dataset statement

`NO DATASET IMAGE, CLASS, OR SPLIT WAS MODIFIED BY STEP 4.`
