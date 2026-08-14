# Classificador FFT Q1.15

Este diretório contém os artefatos do classificador de estado do motor treinado a partir das quatro entradas de aceleração em Q1.15:

- `aceleracao_x_mancal_a`
- `aceleracao_y_mancal_a`
- `aceleracao_x_mancal_b`
- `aceleracao_y_mancal_b`

A cadeia usa janelas não sobrepostas de 64 amostras, aplica a FFT de referência em `03.Reference/fft.py`, requantiza a saída da FFT para Q1.15 e extrai 256 features de magnitude aproximada:

```text
feature = min(32767, abs(real_q15) + abs(imag_q15))
```

![Arvore e features do classificador](classifier_tree_features.svg)

## Classes

| Saída | Classe |
|---:|---|
| 0 | `operacao_normal` |
| 1 | `desalinhamento` |
| 2 | `desbalanceamento` |
| 3 | `desgaste_rolamento` |

## Como Rodar

Execute os comandos a partir da raiz do projeto.

Instale as dependências no ambiente virtual:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Rode um teste rápido com poucas janelas por classe:

```bash
.venv/bin/python 03.Reference/ml_classifier.py \
  --max-windows-per-class 200 \
  --cv-folds 3 \
  --output-dir /tmp/ml_classifier_smoke
```

Rode o treinamento padrão:

```bash
.venv/bin/python 03.Reference/ml_classifier.py
```

Por padrão, o treinamento usa:

- `--dataset 07.Datasets/processed/motor_measurements_q15.parquet`
- `--window-size 64`
- `--max-depth 5`
- `--min-samples-leaf 16`
- `--max-windows-per-class 20000`
- `--test-size 0.15`
- `--val-size 0.15`
- `--cv-folds 5`
- `--output-dir 03.Reference/artifacts/ml_classifier`

## Artefatos

| Arquivo | Uso |
|---|---|
| `model.joblib` | Modelo `DecisionTreeClassifier` treinado para uso em Python. |
| `metrics.json` | Acurácia, matriz de confusão, relatório por classe, validação cruzada e contagem das amostras. |
| `tree_q15.json` | Estrutura da árvore pronta para portar para RTL: nós, features, limiares Q1.15 e folhas com classe `0..3`. |
| `feature_map.csv` | Mapa de cada feature para canal de entrada e bin da FFT. |
| `classifier_tree_features.svg` | Imagem com a extração de features e a árvore treinada. |

## Resultado Atual

O treinamento padrão gerou 20.000 janelas por classe, totalizando 80.000 amostras balanceadas.

Métricas principais:

- validação cruzada média: `0.7885`
- validação: `0.7902`
- teste: `0.7852`
- features: `256`
- profundidade da árvore: `5`
- nós da árvore: `35`

Esses valores são apenas uma primeira referência. A prioridade desta versão é manter a cadeia simples para futura implementação em RTL e ponto fixo.

## Formato para RTL

Cada nó interno de `tree_q15.json` segue a regra:

```text
if feature[feature_index] <= threshold_q15:
    next = left_child
else:
    next = right_child
```

Cada folha retorna diretamente a classe inteira `0..3`.

As features são inteiros Q1.15 não negativos entre `0` e `32767`. O classificador não retorna Q1.15; ele retorna apenas o identificador inteiro da classe.
