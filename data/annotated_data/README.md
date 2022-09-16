# Annotated Data


This directory contains the manually edited examples in our test set -- split evenly across Delta-NLI and SNLI.

### Delta-NLI
Annotated data is in `edited_hyp_dnli.csv`. Columns are: 
```
premise: premise of the example, taken from the original SNLI dataset
hypothesis: hypothesis of the example, taken from the original SNLI dataset
edited_hypothesis: manually edited hypothesis crafted to flip the gold label (strengthener -> weakener, weakener -> strengthener)
update: update sentence of the examples (see original Delta-NLI paper for explanation)
update_type: **original** label of the example. The label induced by the edited hypothesis will be the opposite of this (strengthener -> weakener, weakener -> strengthener)
label: integer corresponding to update type (1 = strengthener, 0 = weakener)
id: original example id
dataset: split of defeasible data (always SNLI in our case)
```

### SNLI
Annotated data is in `edited_hyp_snli.csv`. Columns are: 
```
pairID: original SNLI pair ID
premise: premise of the example, taken from the original SNLI dataset
edited_premise: manually edited premise crafted to change the gold label
hypothesis: hypothesis of the example, taken from the original SNLI dataset
gold_label: **original** label of the example.
target_label: label induced by the edited premise
label: integer corresponding to original gold label
```

### Validated Data
`validated_annotated_snli_sample.csv`: Validated data sample for SNLI -- annotated label is in `label_annotated` and agreement should be calculated with `target_label`. <br>
`validated_annotated_dnli_sample.csv`: Validated data sample for DNLI -- annotated label is also in `label_annotated` and agreement should be calculated with flip of `update_type`.

