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
4. Reduce number of training steps to 6k, maybe the reason that convergences are really slow, even after the inital phase of training was super fast, is because the learning rate is still to high, and further convergence needs lower learning rates. So the distillation training runs in a way just waiting for the lower learning rate.
5. Save graph (4.)


Conclusion 2.: After distillation has stopped, with regular training the new model is able to follow the training curve of the 30M to 30M distillation for the whole training. It seems that this cutoff at 3000 is a bit too harsh however, and a more linear transition may be necessary, which will not set back the training process, as it did in this case.
Conclusion 4.: I assume that the loss will just converge faster, as if 12k and 6k steps are equal, sort of, learning rate is indeed the bottleneck
Other Conclusions: Early hidden states distillation supports much higher learning rates, effectively speeding up early training, but is quickly bottle-necked by too high of a learning rate for lower loss values. Maximally effective distillation requires precise manipulation of the learning rate. Transitions between learning rates don't have to be smooth (can be jumps), because it seems that the optimizer AdamW can smoothen that out.
End of day thoughts and guidance for tomorrow: It seems I was technically able to reduce number of iterations it takes to reach target loss by 3x (from 12k steps to 4k steps). And this was done with only 2k distillation steps I believe. Maybe this can be improved even further, but careful learning rate management is necessary. I believe a more automatic learning rate scheduler may be useful, which I need to explore. I should probably implement attention distillation, and then try to replicate these results from hidd distillation with logits and or attention distillation. Then I think I can move on to try some self-distillation techniques. Oh yeah, I should also probably different teacher sizes for logits and attn distillation, to see if they have any effect on the distillation speedup, or if the only thing that matters is validation loss. Furthermore, I should try and compare training with aux head but letting gradients flow back in, to see difference in performance, then that same with distillation from final head to single intermediate head. I should also try training only a couple of layers of a transformer, and then freezing those and appending more, to see how loss behaves and all, info about this could be crucial for any self-distillation hope. Put these all into a list tomorrow and complete the list at least mindlessly! 

---