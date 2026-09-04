# Top Classifier FFT + MDC + ML

Este diretório documenta o fluxo de referência `03.Reference/ml_pipeline.py`
avaliado por `03.Reference/evaluate_datasets.py`.

O fluxo implementado é:

```text
dataset Q1.7 ou Q1.15
  -> LMS desligado
  -> FFT de 64 pontos por canal
  -> bins 0..32
  -> 3 maiores picos por canal
  -> MDC dos indices dos picos
  -> classificador ML
```

## Entrada

O avaliador varre os Parquets processados em:

```text
07.Datasets/processed/*.parquet
```

O formato é inferido pelo tipo dos quatro canais de vibração: `int8` para Q1.7
e `int16` para Q1.15.

Na API direta de `MLPipeline`, o padrão é `data_width=8`, `lms=False` e
`mdc=True`.

Os canais usados são:

```text
aceleracao_x_mancal_a
aceleracao_y_mancal_a
aceleracao_x_mancal_b
aceleracao_y_mancal_b
```

## Features

Para cada janela de 64 amostras:

- O LMS fica desligado no fluxo oficial.
- A FFT usa os bins `0..32`.
- A magnitude de cada bin é saturada para o formato selecionado:

```text
Q1.7:  feature = min(127, abs(real_q17) + abs(imag_q17))
Q1.15: feature = min(32767, abs(real_q15) + abs(imag_q15))
```

O vetor final tem 140 features:

```text
4 canais x 33 bins FFT = 132 features
4 canais x 2 features MDC = 8 features
total = 140 features
```

As features MDC por canal são:

```text
f0
valid
```

## Comandos

Execute os comandos a partir da raiz do projeto.

Rodar um teste rápido com 200 janelas por classe:

```bash
.venv/bin/python 03.Reference/evaluate_datasets.py \
  --max-windows-per-class 200 \
  --cv-folds 3 \
  --output-dir /tmp/top_classifier_smoke_200
```

Rodar o treinamento completo padrão:

```bash
.venv/bin/python 03.Reference/evaluate_datasets.py
```

Rodar o treinamento completo escolhendo explicitamente o diretório de saída:

```bash
.venv/bin/python 03.Reference/evaluate_datasets.py \
  --output-dir 03.Reference/artifacts/dataset_evaluation
```

Alterar os parâmetros do MDC:

```bash
.venv/bin/python 03.Reference/evaluate_datasets.py \
  --fs-hz 6400 \
  --min-k 2
```

## Artefatos Gerados

O treinamento gera:

| Arquivo | Uso |
|---|---|
| `model.joblib` | Modelo `DecisionTreeClassifier` treinado para uso em Python. |
| `metrics.json` | Acurácia, matriz de confusão, relatório por classe e validação cruzada. |
| `tree_q1_7.json` ou `tree_q1_15.json` | Árvore exportada em formato simples para portar para RTL. |
| `feature_map.csv` | Mapa das 140 features: FFT e MDC. |

## Acurácia Obtida

Foi executado um smoke test com 200 janelas por classe, totalizando 800 janelas balanceadas:

```bash
.venv/bin/python 03.Reference/evaluate_datasets.py \
  --max-windows-per-class 200 \
  --cv-folds 3 \
  --output-dir /tmp/top_classifier_smoke_200
```

Esse comando escreve `comparison.json`, `comparison.csv`, `pipeline_config.json`,
`metrics.json`, `model.joblib` e a árvore quantizada de cada dataset válido.

## Parâmetros Padrão

```text
--fs-hz 6400
--min-k 2
--data-width 8
--max-depth 5
--min-samples-leaf 16
--max-windows-per-class None
--test-size 0.15
--val-size 0.15
--cv-folds 5
--output-dir 03.Reference/artifacts/dataset_evaluation
```
