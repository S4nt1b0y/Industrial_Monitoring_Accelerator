# Caminho da CNN

Classificador convolucional de falhas, acoplado ao `top_wrapper` pela
chave `ml_swith`. Tudo que ele precisa está nesta pasta, incluindo cópias
próprias dos blocos compartilhados (`common/`, `decimator/`, `fft/`,
`spectrogram/`), para não se misturar com a FFT do caminho da árvore de
decisão — os nomes de módulo são diferentes e os dois convivem sem
conflito.

## O que muda fora desta pasta

Só `04.RTL/top/top_wrapper.v`, em dois pontos:

1. `UART_BAUD_RATE` deixou de ser `localparam` e virou parâmetro, com o
   mesmo valor padrão de antes (9600). Nada muda para quem já usava.
2. A instância `u_cnn`, que estava comentada, foi preenchida.

O port list comentado que existia antes não podia ser usado: ele
referenciava `acc_x_a`, `acc_x_b`, `acc_y_a` e `acc_y_b`, que não existem
no arquivo, e o buffer de quadros só expõe um bloco de canal por vez. A
instância real recebe `sample_block`/`frame_channel`, como o
`u_ml_pipeline` logo acima.

Isso também resolve um problema que já existia: com a instância
comentada, `cnn_ready_o` ficava sem driver e `ml_swith = 1` travava a
ingestão em `z`. Agora os dois caminhos funcionam.

## Cadeia

```
sample_block (64 amostras Q1.15)
  -> block_feeder      desdobra em stream serial, marcado por canal
  -> decimator_x64     /64 em dois estagios polifase
  -> prescale          recupera o headroom que a decimacao consome
  -> spectrogram       32 colunas x 32 bins
  -> normalize         z-score, Q1.15 -> Q8.8
  -> cnn_core          conv/ReLU/pool x8 filtros, densa, argmax
  -> class_o
```

## Três coisas a saber antes de mexer

**Canais.** O transmissor manda x_A, y_A, x_B, y_B nessa ordem, então as
tags 0 e 1 do buffer são o mancal A, que é o par com que a rede foi
treinada. As tags 2 e 3 são consumidas e descartadas, com `ready_o`
sempre alto — se ele baixasse, a ingestão travaria e `overflow_o`
acenderia. Atenção: os `localparam` em `ml_pipeline_fsm.v` chamam a tag 1
de `CH_X_B`, o que contradiz o transmissor. Aqui seguimos a ordem do fio.
Vale o time decidir qual dos dois está errado.

**Numeração de classe.** Esta rede ordena as classes como normal,
desbalanceamento, desalinhamento, desgaste; o decode de LED do
`top_wrapper` espera normal, desalinhamento, desbalanceamento, desgaste.
As classes 1 e 2 são trocadas dentro do `cnn.v`, então nada fora dele
precisa mudar.

**Tempo por classificação.** O espectrograma precisa de 576 amostras
decimadas por canal, ou seja 36.864 amostras brutas, entregues 64 por
vez: 576 quadros de 512 bytes. A 9600 baud isso passa de cinco minutos; a
115200 fica em ~26 s. Para a demo da CNN, sintetize com
`UART_BAUD_RATE = 115200`.

## Pesos

`weights/*.hex`, Q8.8, gerados por
`03.Reference/tools/export_cnn_weights_hex.py` do projeto de origem. Os
caminhos são parâmetros de `cnn.v` e por padrão assumem que o diretório
de trabalho da ferramenta é a raiz do projeto. Se a sua ferramenta roda
de outro lugar, sobrescreva `KERNEL_FILE`, `CBIAS_FILE`, `DBIAS_FILE` e
`DW0_FILE`..`DW3_FILE`.

As constantes `GAIN = 287` e `OFFSET = 116` em `normalize.v` dependem dos
pesos treinados e do valor do prescale. Retreinar a rede exige regerar os
`.hex` **e** essas duas constantes juntas.

## Testes

```
cd 05.Sim/cnn
make all       # 68 checks, rapido
make tb_cnn    # aceitacao: sinal real -> classe correta, alguns minutos
```

O `tb_cnn` alimenta uma janela real de `0Nm_BPFI_03` (desgaste de
rolamento) pela mesma interface de bloco que o buffer de UART apresenta,
incluindo os canais 2 e 3 que a rede descarta, e confere a classe.
