## Current goals

## Distill different teacher comparison study
Comparison of using different teachers on the same 30M param architecture, to see
how to they differ in their earlier training curves. They will probably
all converge to the same resulting loss.

### 30M distill into 30M baseline
When distilling from a fully trained 30M model, into a newly initialized 30M model with the same
architecture, the results are as shown:
![30M Distill into 30M](readme_files/30M_distill_30M.PNG)

---

### 70M distill into 30M case
Logits distilling will be easy, but hidden state will probably not be too 
successful.

1. Try to reproduce baseline first, for a few thousand steps.
2. Perform 70M into 30M run.


---

### 30M intermediate checkpoint distill into 30M 
Using a not fully trained checkpoint of the 30M model, distill into freshly
initialized model.
