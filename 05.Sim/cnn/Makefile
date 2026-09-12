IVERILOG ?= iverilog
VVP      ?= vvp

RTL       := ../../04.RTL/cnn
RTL_COM   := $(RTL)/common
RTL_DEC   := $(RTL)/decimator
RTL_FFT   := $(RTL)/fft
RTL_SPEC  := $(RTL)/spectrogram

IFLAGS := -g2005

CORE_SRCS := $(RTL_COM)/mac.v \
             $(RTL_COM)/mac_array.v \
             $(RTL_COM)/ram.v \
             $(RTL)/weight_rom.v \
             $(RTL)/bundle_rom.v \
             $(RTL)/normalize_datapath.v \
             $(RTL)/normalize_ctrl.v \
             $(RTL)/normalize.v \
             $(RTL)/conv2d_datapath.v \
             $(RTL)/conv2d_ctrl.v \
             $(RTL)/conv2d_top.v \
             $(RTL)/maxpool_datapath.v \
             $(RTL)/maxpool_ctrl.v \
             $(RTL)/maxpool.v \
             $(RTL)/dense_datapath.v \
             $(RTL)/dense_ctrl.v \
             $(RTL)/dense.v \
             $(RTL)/argmax.v \
             $(RTL)/cnn_core_datapath.v \
             $(RTL)/cnn_core_ctrl.v \
             $(RTL)/cnn_core.v

PATH_SRCS := $(CORE_SRCS) \
             $(RTL_COM)/done_latch.v \
             $(RTL_DEC)/fir_coef_rom.v \
             $(RTL_DEC)/fir_decim_datapath.v \
             $(RTL_DEC)/fir_decim_ctrl.v \
             $(RTL_DEC)/fir_decim.v \
             $(RTL_DEC)/decimator_x64.v \
             $(RTL_FFT)/twiddle_rom.v \
             $(RTL_FFT)/fft_butterfly.v \
             $(RTL_FFT)/fft_datapath.v \
             $(RTL_FFT)/fft_ctrl.v \
             $(RTL_FFT)/fft.v \
             $(RTL_FFT)/isqrt.v \
             $(RTL_FFT)/magnitude_datapath.v \
             $(RTL_FFT)/magnitude_ctrl.v \
             $(RTL_FFT)/magnitude.v \
             $(RTL_SPEC)/prescale.v \
             $(RTL_SPEC)/spectrogram_datapath.v \
             $(RTL_SPEC)/spectrogram_ctrl.v \
             $(RTL_SPEC)/spectrogram.v \
             $(RTL)/block_feeder_datapath.v \
             $(RTL)/block_feeder_ctrl.v \
             $(RTL)/block_feeder.v \
             $(RTL)/cnn_seq_ctrl.v \
             $(RTL)/cnn.v

.PHONY: help tb_weight_rom tb_conv2d tb_maxpool tb_argmax tb_dense tb_block_feeder \
	tb_normalize tb_prescale tb_cnn_core tb_cnn all clean

help:
	@echo "Alvos disponiveis:"
	@echo "  tb_weight_rom   ROM de pesos"
	@echo "  tb_conv2d       Convolucao 3x3 com padding (CIN=1 e CIN=2)"
	@echo "  tb_maxpool      Max-pooling 2x2 passo 2 (com negativos)"
	@echo "  tb_argmax       Argmax de 4 logits (empates e negativos)"
	@echo "  tb_dense        Camada densa 8->4 em Q8.8"
	@echo "  tb_block_feeder Bloco de 64 amostras -> stream serial"
	@echo "  tb_normalize    Ponte z-score Q1.15 -> Q8.8"
	@echo "  tb_prescale     Deslocamento saturante antes da FFT"
	@echo "  tb_cnn_core     Nucleo completo 32x32x2 -> classe"
	@echo "  tb_cnn          Aceitacao: sinal real -> classe correta (lento)"
	@echo "  all             Roda todos menos o de aceitacao"
	@echo "  clean           Remove os .vvp compilados"

tb_weight_rom:
	$(IVERILOG) $(IFLAGS) -o tb_weight_rom.vvp $(RTL)/weight_rom.v tb_weight_rom.v
	$(VVP) tb_weight_rom.vvp

tb_conv2d:
	$(IVERILOG) $(IFLAGS) -o tb_conv2d.vvp \
		$(RTL_COM)/mac.v $(RTL_COM)/mac_array.v $(RTL_COM)/ram.v \
		$(RTL)/conv2d_datapath.v $(RTL)/conv2d_ctrl.v $(RTL)/conv2d_top.v \
		tb_conv2d.v
	$(VVP) tb_conv2d.vvp

tb_maxpool:
	$(IVERILOG) $(IFLAGS) -o tb_maxpool.vvp \
		$(RTL_COM)/ram.v \
		$(RTL)/maxpool_datapath.v $(RTL)/maxpool_ctrl.v $(RTL)/maxpool.v \
		tb_maxpool.v
	$(VVP) tb_maxpool.vvp

tb_argmax:
	$(IVERILOG) $(IFLAGS) -o tb_argmax.vvp $(RTL)/argmax.v tb_argmax.v
	$(VVP) tb_argmax.vvp

tb_dense:
	$(IVERILOG) $(IFLAGS) -o tb_dense.vvp \
		$(RTL_COM)/mac.v $(RTL_COM)/mac_array.v $(RTL_COM)/ram.v \
		$(RTL)/weight_rom.v \
		$(RTL)/dense_datapath.v $(RTL)/dense_ctrl.v $(RTL)/dense.v \
		$(RTL)/argmax.v \
		tb_dense.v
	$(VVP) tb_dense.vvp

tb_block_feeder:
	$(IVERILOG) $(IFLAGS) -o tb_block_feeder.vvp \
		$(RTL)/block_feeder_datapath.v $(RTL)/block_feeder_ctrl.v \
		$(RTL)/block_feeder.v \
		tb_block_feeder.v
	$(VVP) tb_block_feeder.vvp

tb_normalize:
	$(IVERILOG) $(IFLAGS) -o tb_normalize.vvp \
		$(RTL_COM)/mac.v $(RTL_COM)/mac_array.v $(RTL_COM)/ram.v \
		$(RTL)/normalize_datapath.v $(RTL)/normalize_ctrl.v $(RTL)/normalize.v \
		tb_normalize.v
	$(VVP) tb_normalize.vvp

tb_prescale:
	$(IVERILOG) $(IFLAGS) -o tb_prescale.vvp $(RTL_SPEC)/prescale.v tb_prescale.v
	$(VVP) tb_prescale.vvp

tb_cnn_core:
	$(IVERILOG) $(IFLAGS) -o tb_cnn_core.vvp $(CORE_SRCS) tb_cnn_core.v
	$(VVP) tb_cnn_core.vvp

# Acceptance test: reads vectors/samples_ch*.hex and the weights in
# 04.RTL/cnn/weights. Takes a couple of minutes, so it is not in `all`.
tb_cnn:
	$(IVERILOG) $(IFLAGS) -o tb_cnn.vvp $(PATH_SRCS) tb_cnn.v
	$(VVP) tb_cnn.vvp

all: tb_weight_rom tb_conv2d tb_maxpool tb_argmax tb_dense tb_block_feeder \
	tb_normalize tb_prescale tb_cnn_core

clean:
	rm -f *.vvp
