# Teste UART/FPGA com ML

Este guia descreve como enviar uma janela de 64 amostras de um arquivo Parquet
Q1.15 para a FPGA usando o script `03.Reference/uart_tx_parquet.py`. A FPGA deve
estar configurada com o `top_wrapper.v`, recebendo os bytes pela UART do CP2102 e
processando os dados pelo pipeline de ML.

## Objetivo

O fluxo esperado e:

1. configurar a FPGA com o wrapper atual;
2. conectar o CP2102 USB-TTL ao pino UART RX da FPGA;
3. rodar o script no computador;
4. enviar exatamente uma janela de dados;
5. aguardar a classificacao no hardware;
6. observar um LED de classe aceso.

Por padrao, o script envia as primeiras 64 linhas do dataset
`07.Datasets/processed/motor_measurements_q15.parquet`.

## Arquivos envolvidos

| Arquivo | Funcao |
|---|---|
| `03.Reference/uart_tx_parquet.py` | Le 64 amostras Q1.15 e envia pela UART. |
| `07.Datasets/processed/motor_measurements_q15.parquet` | Dataset padrao, com canais `int16`. |
| `04.RTL/top/top_wrapper.v` | Integra UART, buffer, pipeline ML e LEDs. |
| `04.RTL/uart/uart_rx.v` | Receptor UART 8N1. |
| `04.RTL/uart/uart_frame_buffer.v` | Agrupa bytes em blocos de 64 amostras por canal. |
| `05.Sim/top/tb_top_wrapper.v` | Testbench do caminho UART para ML. |

## Preparar ambiente Python

Na raiz do repositorio:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Para conferir se o `pyserial` esta disponivel:

```bash
.venv/bin/python - <<'PY'
import serial
print(serial.VERSION)
PY
```

Para ver as opcoes do script:

```bash
.venv/bin/python 03.Reference/uart_tx_parquet.py --help
```

## Preparar FPGA

Use o `04.RTL/top/top_wrapper.v` como topo, ou instancie o wrapper em um topo da
sua placa mantendo estes parametros e sinais:

| Item | Valor esperado |
|---|---|
| `DATA_WIDTH` | `16` |
| `N` | `64` |
| `CLK_FREQ_HZ` | frequencia real do clock da placa |
| `UART_BAUD_RATE` | `9600` |
| `ml_swith` | `0` para selecionar ML |
| `rst_n` | `1` depois do reset |
| `uart_rx_i` | entrada ligada ao TX do CP2102 |

Mapeie tambem os LEDs de saida:

| Sinal | Classe indicada |
|---|---|
| `Led_Normal` | classe `0`, operacao normal |
| `Led_disalaighn` | classe `1`, desalinhamento |
| `Led_Unbalaced` | classe `2`, desbalanceamento |
| `Led_desgaste` | classe `3`, desgaste de rolamento |

Importante: mantenha `ml_swith = 0`. O caminho CNN esta comentado no wrapper
atual, entao `ml_swith = 1` nao deve ser usado neste teste.

## Conexao CP2102

Com a placa desligada, faca as conexoes:

| CP2102 | FPGA |
|---|---|
| `TXD` | pino mapeado para `uart_rx_i` |
| `GND` | `GND` da placa |

Nao e necessario ligar o `RXD` do CP2102 para este teste, porque o script apenas
transmite dados para a FPGA. Garanta que o nivel TTL do CP2102 e compativel com a
FPGA, preferencialmente 3.3 V.

Depois de conectar, configure a FPGA e tire o circuito de reset.

## Descobrir porta serial

Com o CP2102 conectado ao computador:

```bash
.venv/bin/python 03.Reference/uart_tx_parquet.py --list-ports
```

Em Linux, a porta normalmente aparece como `/dev/ttyUSB0` ou `/dev/ttyUSB1`.

Se aparecer erro de permissao ao abrir a porta, adicione seu usuario ao grupo
`dialout` e faca logout/login:

```bash
sudo usermod -a -G dialout "$USER"
```

## Teste sem hardware: dry-run

Antes de abrir a UART, valide a leitura do Parquet e a montagem do payload:

```bash
.venv/bin/python 03.Reference/uart_tx_parquet.py --dry-run
```

Resultado esperado:

```text
Samples per channel: 64
Channels: 4
Payload bytes: 512
Dry run: UART port was not opened.
```

O total de 512 bytes vem de:

```text
4 canais * 64 amostras * 2 bytes = 512 bytes
```

## Envio real para FPGA

Com a FPGA configurada, `ml_swith = 0`, CP2102 conectado e porta serial
identificada:

```bash
.venv/bin/python 03.Reference/uart_tx_parquet.py --port /dev/ttyUSB0
```

O script usa por padrao:

| Opcao | Valor |
|---|---|
| `--dataset` | `07.Datasets/processed/motor_measurements_q15.parquet` |
| `--baud` | `9600` |
| `--offset` | `0` |
| `--count` | `64` |
| `--timeout` | `2.0` |

O formato transmitido e:

1. 64 amostras de `aceleracao_x_mancal_a`;
2. 64 amostras de `aceleracao_y_mancal_a`;
3. 64 amostras de `aceleracao_x_mancal_b`;
4. 64 amostras de `aceleracao_y_mancal_b`.

Cada amostra e `int16` signed, em big-endian: primeiro MSB, depois LSB. O CP2102
transforma esses bytes em bits UART 8N1 na linha fisica.

## Enviar janelas seguintes

O script envia uma janela e encerra. Para mandar novos dados, rode o script de
novo com outro offset:

```bash
.venv/bin/python 03.Reference/uart_tx_parquet.py --port /dev/ttyUSB0 --offset 64
.venv/bin/python 03.Reference/uart_tx_parquet.py --port /dev/ttyUSB0 --offset 128
.venv/bin/python 03.Reference/uart_tx_parquet.py --port /dev/ttyUSB0 --offset 192
```

Use offsets multiplos de 64 para manter janelas alinhadas.

## Comportamento esperado

Depois do envio:

1. o `uart_rx` recebe bytes UART 8N1 a 9600 baud;
2. o `uart_frame_buffer` monta 4 blocos de 64 amostras;
3. o `ml_pipeline` processa um canal por vez;
4. ao final do quarto canal, o classificador gera `class_o`;
5. o `top_wrapper` acende o LED correspondente.

Exatamente um LED de classe deve ficar aceso quando a classificacao valida
chegar. O LED fica retido ate ocorrer uma destas condicoes:

| Evento | Efeito |
|---|---|
| nova classificacao valida | atualiza para a nova classe |
| reset da FPGA | apaga os LEDs |
| overflow do buffer UART | apaga os LEDs |

O script nao recebe uma resposta da FPGA. A confirmacao visual deste teste e o
LED aceso na placa.

## Problemas comuns

| Sintoma | Possivel causa | Acao |
|---|---|---|
| Nenhuma porta aparece em `--list-ports` | CP2102 desconectado ou driver ausente | reconectar o USB e verificar `dmesg`/driver |
| Erro de permissao em `/dev/ttyUSB0` | usuario sem acesso a serial | adicionar ao grupo `dialout` e fazer logout/login |
| Nenhum LED acende | `ml_swith` em `1`, reset ativo ou pino UART incorreto | usar `ml_swith = 0`, liberar `rst_n`, revisar constraints |
| LEDs apagam apos envio | `frame_overflow` | enviar apenas uma janela por vez e evitar bytes extras |
| Dataset Q1.7 rejeitado | colunas `int8` | usar Parquet Q1.15, como `motor_measurements_q15.parquet` |
| Classificacao nao muda | mesma janela enviada de novo | usar `--offset 64`, `--offset 128`, etc. |

## Validacao por simulacao

O testbench `05.Sim/top/tb_top_wrapper.v` ja exercita o caminho UART para ML: ele
envia 4 blocos de 64 amostras por UART, verifica que o ML aceitou 4 blocos e
confirma que o pipeline produziu uma saida valida.

Um comando direto com `iverilog`:

```bash
iverilog -g2012 -o /tmp/tb_top_wrapper.vvp \
  05.Sim/top/tb_top_wrapper.v \
  04.RTL/top/top_wrapper.v \
  04.RTL/uart/uart_rx.v \
  04.RTL/uart/uart_frame_buffer.v \
  04.RTL/top/ml_pipeline.v \
  04.RTL/top/ml_pipeline_fsm.v \
  04.RTL/fft/fftu_dif.v \
  04.RTL/fft/butterfly_dif.v \
  04.RTL/fft/twiddle_lut.v \
  04.RTL/fft/addr_gen.v \
  04.RTL/mdc/mdc.v \
  04.RTL/ml_classifier/ml_classifier.v
vvp /tmp/tb_top_wrapper.vvp
```

Resultado esperado:

```text
All top_wrapper UART/ML route tests passed.
```
