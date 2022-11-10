# Extractive Question Answering

- task of extracting an answer from a text given a question
- SQuAD dataset


## QuestionaAnsweringEvaluator

https://huggingface.co/docs/evaluate/v0.3.0/en/package_reference/evaluator_classes#evaluate.QuestionAnsweringEvaluator

```py
from evaluate import evaluator
from datasets import load_dataset
task_evaluator = evaluator("question-answering")
data = load_dataset("squad_v2", split="validation[:2]")
results = task_evaluator.compute(
    model_or_pipeline="mrm8488/bert-tiny-finetuned-squadv2",
    data=data,
    metric="squad_v2",
    squad_v2_format=True,
)
```

https://colab.research.google.com/gist/nclskfm/960d28565395928f6df97894e21541b7/evaluation-square-models.ipynb

## Metric: squad_v2

- 100.000 questions by SQuAD v.1.1 and 50.000 unanswerable questions
  > systems must not only answer questions when possible, but also determine when no answer is supported by the paragraph and abstain from answering

### Output

- `'exact'`: Exact match (the normalized answer exactly match the gold answer) (see the exact_match metric (forthcoming))
- `'f1'`: The average F1-score of predicted tokens versus the gold answer (see the F1 score metric)
- `'total'`: Number of scores considered
- `'HasAns_exact'`: Exact match (the normalized answer exactly match the gold answer)
- `'HasAns_f1'`: The F-score of predicted tokens versus the gold answer
- `'HasAns_total'`: How many of the questions have answers
- `'NoAns_exact'`: Exact match (the normalized answer exactly match the gold answer)
- `'NoAns_f1'`: The F-score of predicted tokens versus the gold answer
- `'NoAns_total'`: How many of the questions have no answers
- `'best_exact'` : Best exact match (with varying threshold)
- `'best_exact_thresh'`: No-answer probability threshold associated to the best exact match
- `'best_f1'`: Best F1 score (with varying threshold)
- `'best_f1_thresh'`: No-answer probability threshold associated to the best F1

### Values form popular papers

- SQuAD v2 paper
  - F1 score: 66.3%
  - Exact Match score: 63.4%.
- Human Performance
  - F1 score 89.5%
  - Exact Match score86.9%.
