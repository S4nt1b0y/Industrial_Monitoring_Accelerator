# Top Classifier LMS + FFT + MDC + ML

Este diretório documenta o uso do modelo de referência `03.Reference/top_classifier.py`.

O fluxo implementado é:

```text
dataset Q1.15
  -> LMS por canal com referencia atrasada em 1 amostra
  -> FFT de 64 pontos por canal
  -> bins 0..32
  -> 3 maiores picos por canal
  -> MDC dos indices dos picos
  -> classificador ML
```

## Entrada

Por padrão, o script usa:

```text
07.Datasets/processed/motor_measurements_q15.parquet
```

Esse dataset foi escolhido porque teve a melhor acurácia de teste na comparação Q1.15 já existente em `03.Reference/artifacts/ml_classifier/comparison_q15.json`.

Os canais usados são:

```text
aceleracao_x_mancal_a
aceleracao_y_mancal_a
aceleracao_x_mancal_b
aceleracao_y_mancal_b
```

## Features

Para cada janela de 64 amostras:

- O LMS é aplicado independentemente em cada canal.
- A referência do LMS é o mesmo canal atrasado em 1 amostra.
- A saída usada na FFT é `y`, ou seja, a saída filtrada do LMS.
- A FFT usa os bins `0..32`.
- A magnitude de cada bin é calculada como:

```text
feature = min(32767, abs(real_q15) + abs(imag_q15))
```

O vetor final tem 144 features:

```text
4 canais x 33 bins FFT = 132 features
4 canais x 3 features MDC = 12 features
total = 144 features
```

As features MDC por canal são:

```text
mdc_k0
mdc_f0_hz
mdc_result_valid
```

## Comandos

Execute os comandos a partir da raiz do projeto.

Listar datasets Q1.15 válidos:

```bash
.venv/bin/python 03.Reference/top_classifier.py --list-datasets
```

Rodar um teste rápido com 200 janelas por classe:

```bash
.venv/bin/python 03.Reference/top_classifier.py \
  --dataset motor_measurements_q15.parquet \
  --max-windows-per-class 200 \
  --cv-folds 3 \
  --output-dir /tmp/top_classifier_smoke_200
```

Rodar o treinamento completo padrão:

```bash
.venv/bin/python 03.Reference/top_classifier.py
```

Rodar o treinamento completo escolhendo explicitamente o diretório de saída:

```bash
.venv/bin/python 03.Reference/top_classifier.py \
  --dataset motor_measurements_q15.parquet \
  --output-dir 03.Reference/artifacts/top_classifier/motor_measurements_q15
```

Alterar o atraso da referência do LMS:

```bash
.venv/bin/python 03.Reference/top_classifier.py \
  --lms-delay 1
```

Alterar os parâmetros do MDC:

```bash
.venv/bin/python 03.Reference/top_classifier.py \
  --fs-hz 6400 \
  --min-k 2
```

## Artefatos Gerados

O treinamento gera:

| Arquivo | Uso |
|---|---|
| `model.joblib` | Modelo `DecisionTreeClassifier` treinado para uso em Python. |
| `metrics.json` | Acurácia, matriz de confusão, relatório por classe e validação cruzada. |
| `tree_q15.json` | Árvore exportada em formato simples para portar para RTL. |
| `feature_map.csv` | Mapa das 144 features: FFT e MDC. |

## Acurácia Obtida

Foi executado um smoke test com 200 janelas por classe, totalizando 800 janelas balanceadas:

```bash
.venv/bin/python 03.Reference/top_classifier.py \
  --dataset motor_measurements_q15.parquet \
  --max-windows-per-class 200 \
  --cv-folds 3 \
  --output-dir /tmp/top_classifier_smoke_200
```

Resultado obtido:

```text
Validation accuracy: 0.7438
Test accuracy: 0.7250
Feature count: 144
```

Esses valores são de teste rápido. Para obter a acurácia final do modelo completo, rode o treinamento padrão sem limitar `--max-windows-per-class`; o padrão usa 20.000 janelas por classe.

## Parâmetros Padrão

```text
--dataset motor_measurements_q15.parquet
--window-size 64
--lms-delay 1
--fs-hz 6400
--min-k 2
--max-depth 5
--min-samples-leaf 16
--max-windows-per-class 20000
--test-size 0.15
--val-size 0.15
--cv-folds 5
--output-dir 03.Reference/artifacts/top_classifier/<dataset_stem>
```
