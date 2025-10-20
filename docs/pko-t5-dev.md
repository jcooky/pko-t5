# pko-T5 모델 개발기

> 파우스트에서 한국어 코퍼스를 활용하여 T5 를 개발했던 경험기

---

## Introduction

이번에 PAUST에서 한국어 기반 Open-domain QA 를 개발하였고 이를 T5로 쉽게 개발에 성공했습니다.

현재 Open-domain QA 를 만드는데에 있어 facebook 의 DPR 과 FiD를 활용하여 만드려고 하는 시도들이 많았습니다.

하지만 DPR 을 하기 위한 한국어 기반 사전학습모델은 [monologg/koBERT](https://huggingface.co/monologg/kobert), [klue/roberta-base](https://huggingface.co/klue/roberta-base) 등이 있었지만 [FiD](https://github.com/facebookresearch/FiD) 를 하는데에 있어서 encoder only 모델이 아닌 encoder-decoder 모델이 필요합니다.

encoder-decoder 모델에는 대표적으로 T5 와 BART 가 있습니다. PAUST 에서 한국어 기반의 encoder-decoder 모델을 활용해서 실험을 진행하는데 있어 아래 2가지의 조건이 필요했습니다.

1. BBPE 기반으로 tokenizer 를 만들어서 특수문자도 받을 수 있게 하자.
2. [T5 v1.1](https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md#t511)의 모델을 기반으로 하자.

위 조건들을 충족시키고자 한국어 말뭉치와 transformers 라이브러리를 활용하여 T5 모델 학습에 도전해보았습니다.

tokenizers 라이브러리 덕분에 BBPE 는 쉽게 만들었지만, T5 모델 학습은 기존에는 tensorflow 로 구현 된 것을 저희가 쉽게 transformers 의 파이토치 모델을 활용하여 만들었습니다.

먼저, T5 는 Fig.1 에서 보듯이 text-to-text 라는 포맷으로 모든 데이터셋의 포맷을 통일합니다. 그렇기에 여러가지 NLP task 에도 적용이 가능하여 활용도가 매우 높은 모델입니다.

![Fig.1 Text-to-text 를 수행하는 T5 의 예시](https://1.bp.blogspot.com/-o4oiOExxq1s/Xk26XPC3haI/AAAAAAAAFU8/NBlvOWB84L0PTYy9TzZBaLf6fwPGJTR0QCLcBGAsYHQ/s1600/image3.gif)

*Fig.1 Text-to-text 를 수행하는 T5 의 예시*

이러한 점을 바탕으로 번역, 텍스트분류, 질의응답 등 여러가지 Task에 적용이 가능하며 응용 또한 무궁무진합니다.

이렇게 처음부터 사전학습을 만드는 것은 많은 GPU 리소스를 요구합니다. 저희 PAUST 같은 경우 한정된 GPU 리소스 상에서 여러가지 태스크를 학습해야했습니다. T5 모델 같은 경우 사전학습된 지식을 기반으로 미세한 튜닝(Fine-tuning) 만 하면 되었기에 이러한 점에서 회사에서 큰 장점으로 다가왔습니다.

참고로 T5 v1.1 은 T5 와 아래와 같은 다른 점은 [링크](https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md#t511)에 설명이 되어있습니다.

---

## Implementation

처음에는 이걸 어떻게 만들까? 라는 생각을 했었습니다. 다행히도 모두의 말뭉치와 나무위키, 위키피디아 등등 실제로 쓸만한 한국어 데이터가 많았습니다. 그래서 봤더니 적어도 T5 v1.1 모델 같은 경우는 unsupervised learning 만 하면 되었기에 원래 논문에 있는 multitask-learning 은 배제하고 학습할 수 있어 편하게 진행할 수 있었습니다.

실제 학습의 시작은 [huggingface/transformers](https://github.com/huggingface/transformers) 에서 예제로 제공해준 [run_mlm_t5_flax.py](https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py) 에서 시작을 했지만 여기서 제공해주는건 정적으로 몇개의 비율로 T5의 span 을 만들어낼것인지 결정하기에 논문대로 랜덤하게 동적으로 마스킹을 하고 마스킹된 토큰들을 merge 하게끔 만들었습니다.

```python
def _fill_in_the_blank(self, words: List[int]):
    """
    input:
    {
      words: [My, name, is, T5, .]
    }
    output:
    {
      inputs: [<extra_id_1>, name, is, <extra_id_2>, .]
      outputs: [My, <extra_id_1>, T5, .]
    }
    """
    mask_id = -1

    min_prob = 1 / (len(words) + 1)
    max_prob = 1 / 2
    inputs = copy.deepcopy(words)
    targets = words
    for i in range(len(words)):
        prob = random.random()
        if min_prob < prob < max_prob:
            inputs[i] = mask_id
        else:
            targets[i] = mask_id

    def merge_mask(words_):
        mask_spans = []
        begin, end = None, None
        for i, w in enumerate(words_):
            if w == mask_id:
                if begin is None:
                    begin = i
                end = i + 1
            else:
                if end is not None:
                    mask_spans.append((begin, end))
                    begin, end = None, None
        if begin is not None and end is not None:
            mask_spans.append((begin, end))

        new_words_ = []
        last_offset = 0
        assert len(mask_spans) <= len(self.extra_ids), f"mask_spans={len(mask_spans)} is over length of extra_ids"
        for i, (begin, end) in enumerate(mask_spans):
            new_words_ += words_[last_offset:begin]
            new_words_.append(self.extra_ids[i])
            last_offset = end
        new_words_ += words_[last_offset:]

        return new_words_

    inputs = merge_mask(inputs)
    targets = merge_mask(targets)

    return {'inputs': inputs, 'targets': targets}
```

위 코드에서 최소 1개의 token 이 mask 가 되게하되 최대 50%의 token 만 mask가 되게하면 적어도 한개의 mask 가 포함되기에 target 에서 전체가 mask가 되지 않게 할 수 있습니다.

이러한 작업을 T5 에서는 **Span Corruption Task** 라고 하며 아래와 같이 동작하게 됩니다.

```python
SpanCorruptionTask(tokens=['My', 'name', 'is', 'T5', '.']) => {
  inputs: ['<extra_id_1>', 'name', 'is', '<extra_id_2>', '.']
  outputs: ['My', '<extra_id_1>', 'T5', '.']
}
```

### 학습 환경

학습환경은 pre-training 을 위해 **A100 8장**을 활용하여 학습을 진행했고 small/base/large 를 학습하는데 있어 각각 **3일/16일/26일** 정도가 소요되었습니다. 그리고 각 태스크의 Fine-tuning 을 위해 **TPU v3-8** 노드 한개에서 학습을 진행했습니다.

pre-training 을 위한 자세한 코드는 https://github.com/paust-team/pko-t5 에 올려두었습니다.

이렇게 만들어진 T5를 우리는 **PAUST Korea T5**를 줄여서 **pko-t5**라고 명명하였습니다.

---

## Experiments

### NSMC

먼저 NSMC 데이터를 text-to-text 포맷으로 변환하여 실험해보았습니다. 실험 과정에 앞서 pko-t5 를 활용하여 아래와 같은 결과를 얻었습니다.

| Model name | NSMC's Accuracy |
|-----------|----------------|
| KoBART-base* | 90.24 |
| KETI-AIR/ke-t5-small-ko | 89.37 |
| KETI-AIR/ke-t5-base-ko | **91.38** |
| KETI-AIR/ke-t5-large-ko | 90.67 |
| paust/pko-t5-small | 89.54 |
| paust/pko-t5-base | 91.05 |
| paust/pko-t5-large | 91.15 |
| google/mt5-base | 87.63 |

*\*: SKT 에서 발표한 [KoBART-base](https://github.com/SKT-AI/KoBART) 에서 참고.*

NSMC 에서 결과는 ke-t5 보다 살짝 떨어지지만 그래도 오차범위 내의 준수한 성능을 보여줍니다. NSMC 데이터를 가지고 학습을 할때는 text-to-text 의 성능을 위해서 기존에 encoder-only 아키텍처에서 하던 classification 이 아니라 document → label 을 generation 하도록 디자인했습니다.

예를 들면, 아래와 같은 문장에 대해서 BERT 스타일과 text-to-text 스타일은 각각 다르게 처리합니다.

```python
Input: "pko-t5 프로젝트는 정말 멋지고 훌륭하네요."
BERT label: 1 (positive label)
T5 label: "긍정적인 댓글"
```

BERT와 같은 encoder-only 아키텍처는 사전학습이된 transformer block 의 weight 는 그대로 transfer learning 으로 활용할 수 있지만 마지막 classifier layer 는 새로 만들어야 합니다. 이유는 BERT 학습인 MaskedLM 과 classification 의 label 의 수가 다르기 때문입니다.

하지만, T5 는 label 을 자연어로 표현하여 text generation 으로 태스크를 해결하기 위해 text-to-text 라는 방법론을 제안하였습니다. 그 덕분에 T5는 사전학습에 사용했던 classifier layer 을 그대로 사용하게 됩니다. 이 점을 이용해서 multi-task learning 도 가능하며 여러가지 태스크에도 쉽게 적용할 수 있습니다.

### KorQuAD 1.0 (The Korean Question Answering Dataset)

이번 실험은 korquad 데이터셋을 통해 QA 에서 text-to-text 성능을 확인해보겠습니다. 물론 SOTA 에는 미치지 못하는 성능이지만 다른 T5 대비 어느정도의 pko-t5 가 어느정도의 성능 향상이 있는지 확인할 수 있었습니다.

T5는 QA 태스크에서도 BERT 스타일의 context 내에서 span을 찾기보다는 answer를 바로 생성하도록 학습합니다.

| Model name | Exact-match | F1-score |
|-----------|-------------|----------|
| pko-t5-base | 83.32 | 88.30 |
| pko-t5-large | 86.28 | 91.22 |
| mt5-large | 82.05 | 88.20 |

korquad 에서도 어느정도 pko-t5 가 효과가 있었습니다. mt5-large 와 pko-t5-base 간의 성능이 비슷하고 이미 pko-t5-large 에서 mt5-large 보다 높은 성능을 기록했습니다.

### KLUE (Korean Language Understanding Evaluation)

아래 표는 KLUE 의 dev 셋을 가지고 평가한것입니다. Baseline 은 KLUE 논문에서 가장 성능이 좋다고 한 것을 가져왔습니다.

| Model | tc (macro F1) | sts (pearsonr/F1) | nli (acc) | ner (entity-level F1) | re (micro F1) | dp (LAS) | mrc (EM/F1) |
|-------|---------------|-------------------|-----------|----------------------|---------------|----------|-------------|
| Baseline | 87.30 | 93.20/86.13 | 89.50 | 86.06 | 71.06 | 87.93 | 75.26/- |
| **FT** pko-t5-small | 86.21 | 77.99/77.01 | 69.20 | 82.60 | 62.95 | 93.15 | 43.81/46.58 |
| **FT** pko-t5-base | 87.29 | 90.25/83.43 | 79.73 | 87.80 | 72.94 | 97.28 | 61.53/64.74 |
| **FT** pko-t5-large | 87.12 | 92.05/85.24 | 84.96 | 88.18 | 72.26 | 97.60 | 68.01/71.44 |
| **MT** pko-t5-small | 85.85 | 79.12/77.81 | 66.8 | 81.53 | 67.93 | 91.38 | 44.97/48.07 |
| **MT** pko-t5-base | 86.86 | 87.61/81.42 | 75.46 | 86.85 | 71.85 | 96.32 | 61.95/65.06 |
| **MT** pko-t5-large | 87.25 | 91.05/84.58 | 82.16 | 87.63 | 74.78 | 97.33 | 69.18/71.92 |

위 표에서 **FT** 는 단일 태스크 파인튜닝을 말하며, **MT** 는 멀티태스크 파인튜닝을 말합니다. 즉, MT 는 태스크들을 전부 text-to-text 포맷으로 바꾸어 학습을 진행하고 각 태스크 별로 성능 수치를 뽑았습니다.

TC, STS, NLI 는 기존 encoder 기반의 모델들이 더 잘하고 있었습니다. 이 부분은 text-to-text 로 학습했기에 T5 가 부족한걸로 보입니다.

---

## 관련 링크

- GitHub 저장소: https://github.com/paust-team/pko-t5
- Hugging Face 모델: https://huggingface.co/paust
- 원문 출처: https://e45570339af1.ngrok-free.app/pko-t5-dev-diary/
