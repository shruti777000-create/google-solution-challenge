# Fairness Audit Findings — Biased Model

## Dataset
- UCI Adult Income dataset
- Sensitive attributes audited: sex, race

## Results — Original Model

### By Sex
| Metric | Gap | Threshold | Status |
|--------|-----|-----------|--------|
| Demographic Parity | 13.12% | 10% | ❌ FAIL |
| Equal Opportunity | 1.97% | 10% | ✅ PASS |
| FPR Parity | 4.71% | 10% | ✅ PASS |

### By Race
| Metric | Gap | Threshold | Status |
|--------|-----|-----------|--------|
| Demographic Parity | 20.86% | 10% | ❌ FAIL |
| Equal Opportunity | 29.58% | 10% | ❌ FAIL |
| FPR Parity | 9.96% | 10% | ✅ PASS |

## Key Finding
Race is the most significant source of bias in this model.
Qualified people from certain racial groups are being missed
at nearly 30% higher rates than others. This would translate
to real-world discrimination in hiring or lending decisions.
