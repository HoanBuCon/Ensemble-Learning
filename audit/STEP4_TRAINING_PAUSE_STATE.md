# Step 4 Training Stop State

Recorded after the user cancelled the active DenseNet-121 run and directed that the
next session train from the beginning. This file records persisted state only; it does
not promote the partial run to a final paper result.

## Identity

```text
status: CANCELLED — FRESH RESTART REQUIRED; PARTIAL ARTIFACTS PRESERVED
branch: fix/scientific-pipeline-v2
source_commit: 2e97fce4d1c10f52f00fe4af75110fb98004c036
TRAINING_SOURCE_COMMIT: 2e97fce4d1c10f52f00fe4af75110fb98004c036
SESSION_CLOSING_COMMIT: SELF — the commit that first introduces this report; exact SHA is recorded in the final session output
dataset_manifest_sha256: a94426015d83a3b8e9f66a30bddee9e4b376eaf509f166aa522d7a204fe22dc5
```

## Persisted training boundary

```text
model: densenet121
protocol: single_split
current_epoch: NONE — process stopped
interrupted_in_memory_epoch: 10
last_completed_epoch: 9
best_epoch: 9
best_validation_accuracy: 95.71428571428571
best_validation_loss: 0.5205266234162566
early_stop_counter: 0 at the completed epoch-9 boundary
early_stop_counter_serialized: NO

last_checkpoint_path: RESULTS/FINAL_V2/single_split/densenet121/last_model.pth
best_checkpoint_path: RESULTS/FINAL_V2/single_split/densenet121/best_model.pth

config_path: configs/densenet121.yaml
config_sha256: a0217f0d9fb856f880c6cee0eae6c056c3db326f2177ca52d6d05509d216e93f
result_root: RESULTS/FINAL_V2/single_split/densenet121
resume_command: NOT APPLICABLE — user directed a fresh run rather than resume
fresh_restart_command: NOT EXECUTED; use scripts/train.py without --resume only after assigning a fresh, non-conflicting result root
```

Both checkpoint files identify epoch 9 as the saved epoch and best epoch. They contain
model, optimizer, scheduler, checkpoint-selection, and history state. The early-stop
counter is not a serialized checkpoint field. This limitation does not affect the new
instruction because these checkpoints will not be resumed.

## Stop event

Epoch 9 completed validation and both `best_model.pth` and `last_model.pth` were written
at the epoch boundary. Before the interrupt was delivered, the same process entered
epoch 10 and displayed 28 training batches. The process was interrupted during backward
propagation. No epoch-10 checkpoint, history row, validation result, test result, or
final-experiment artifact was written. All epoch-10 in-memory updates were discarded
when the process exited. The latest durable training state is therefore epoch 9.

No additional backbone, OOF fold, ensemble evaluation, final verification, or final
latency benchmark was started. A process inspection after cancellation found no
`scripts/train.py` process and no Python process associated with this workspace.

## Phase state

```text
Step 4 Code Remediation: COMPLETE
Phase S Final Training: CANCELLED; FRESH RESTART REQUESTED FOR NEXT SESSION
Phase T Final Consistency Audit: NOT STARTED
Phase U Final Freeze: NOT STARTED
```

The existing partial DenseNet files remain under the result root above as historical,
non-final artifacts. They were not deleted, moved, renamed, staged, or classified as
paper results. A future fresh run must use a non-conflicting result identity/root before
launch so these historical partial files are not overwritten.

## Dataset integrity statement

`NO DATASET IMAGE, CLASS, OR SPLIT WAS MODIFIED.`
