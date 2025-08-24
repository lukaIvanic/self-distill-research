## Current goals

## Distill different teacher comparison study
Comparison of using different teachers on the same 30M param architecture, to see
how to they differ in their earlier training curves. They will probably
all converge to the same resulting loss.

### 30M distill into 30M baseline
When distilling from a fully trained 30M model, into a newly initialized 30M model with the same
architecture, the results are as shown:
![30M Distill into 30M](readme_files/30M_distill_30M.PNG)

Conclusion: Distillation significantly speeds up training speed (need to check for val loss graphs), with hidd states distillation sticking out as the more effective method. However benefits end soon with this training regime.
---

### 70M distill into 30M case
Logits distilling will be easy, but hidden state will probably not be too 
successful.

1. Try to reproduce logits baseline first, for a few thousand steps.
2. Perform 70M logits distill into 30M run.
3. Save graph

![70M Distill into 30M](readme_files/70M_distill_into_30M_logits.png)

Conclusion: doesn't perform better than having the trained 30M as a teacher. Strangely, in the first half of the training, it follows the training curve of the teacher 70M model quite precisely. After reaching roughly same loss as the 30M final loss, it plateaus.
---

### 30M intermediate checkpoint distill into 30M 
Using a not fully trained checkpoint of the 30M model, distill into freshly
initialized model.

1. Distill from checkpoint
2. Save graph

![30M 6k steps checkpoint distills into 30M](readme_files/30M_6k_checkpoint_distill_into_30M_logits.png)


Conclusion: Offers slightly faster training in the beginning (first 2k) in comparison to using the final checkpoint as a teacher. Later only proves to be a destructive loss interference.
Learned: The teacher is valid ONLY when it's loss is lower than the student's. The loss difference doesn't have to be large, e.g. even if it's 1% better, it can still be used as a teacher. BUT after the student reaches the teacher's loss, the distillation only interferes with training.
---

## Study deactivating teacher loss after a certain number of steps.
Retrying the 30M to 30M experiments, with more sophisticated training strategies.

1. Reproduce hidden state baseline for 30M distill into 30M
2. Use distillation for the first 3k steps, then keep only regular training
3. Save graph (2.)
4. Retry 2. point, but reduce number of training steps, maybe the reason that converges is really slow, even after the inital phase of training was super fast, is because the learning rate is still to high, and convergence needs lower learning rates.
5. Save graph (4.)