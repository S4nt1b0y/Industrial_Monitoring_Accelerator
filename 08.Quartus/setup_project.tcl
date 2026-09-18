# Projeto Quartus da placa: compila o top_wrapper para a DE0-CV
# (Cyclone V, 5CEBA4F23C7) e gera o bitstream.
#
# O RTL e a pinagem (04.RTL/top/pinout.tcl) sao referenciados de
# ../04.RTL; nenhum arquivo e copiado para esta pasta.
#
# Uso (a partir desta pasta):
#   quartus_sh -t setup_project.tcl
#
# O bitstream sai em output_files/top_wrapper.sof.

package require ::quartus::project
package require ::quartus::flow

set project_name "top_wrapper"

if {[project_exists $project_name]} {
    project_open $project_name -revision $project_name
} else {
    project_new $project_name -revision $project_name
}

set_global_assignment -name FAMILY "Cyclone V"
set_global_assignment -name DEVICE 5CEBA4F23C7
set_global_assignment -name TOP_LEVEL_ENTITY top_wrapper
set_global_assignment -name PROJECT_OUTPUT_DIRECTORY output_files
set_global_assignment -name SDC_FILE top_wrapper.sdc

# Taxa da serial na placa: 115200. O padrao do top_wrapper, 9600, e o
# usado pelo tb_top_wrapper. Transmissores:
#   uart_tx_parquet.py --baud 115200
#   uart_tx_cnn.py              (115200 por padrao)
set_parameter -name UART_BAUD_RATE 115200

set rtl "../04.RTL"
set cnn "$rtl/cnn"

# Caminho ML (arvore de decisao)
set fontes_ml [list \
    $rtl/uart/uart_rx.v \
    $rtl/uart/uart_frame_buffer.v \
    $rtl/fft/twiddle_lut.v \
    $rtl/fft/addr_gen.v \
    $rtl/fft/butterfly_dif.v \
    $rtl/fft/fftu_dif.v \
    $rtl/mdc/mdc.v \
    $rtl/ml_classifier/ml_classifier.v \
    $rtl/top/ml_pipeline_fsm.v \
    $rtl/top/ml_pipeline.v \
    $rtl/top/top_wrapper.v \
]

# Caminho CNN, instanciado no top_wrapper como u_cnn. Os pesos vem de
# 04.RTL/cnn/weights, pelo parametro CNN_WEIGHTS_DIR do top_wrapper.
set fontes_cnn [list \
    $cnn/common/mac.v \
    $cnn/common/mac_array.v \
    $cnn/common/ram.v \
    $cnn/common/done_latch.v \
    $cnn/decimator/fir_coef_rom.v \
    $cnn/decimator/fir_decim_datapath.v \
    $cnn/decimator/fir_decim_ctrl.v \
    $cnn/decimator/fir_decim.v \
    $cnn/decimator/decimator_x64.v \
    $cnn/fft/twiddle_rom.v \
    $cnn/fft/fft_butterfly.v \
    $cnn/fft/fft_datapath.v \
    $cnn/fft/fft_ctrl.v \
    $cnn/fft/fft.v \
    $cnn/fft/isqrt.v \
    $cnn/fft/magnitude_datapath.v \
    $cnn/fft/magnitude_ctrl.v \
    $cnn/fft/magnitude.v \
    $cnn/spectrogram/prescale.v \
    $cnn/spectrogram/spectrogram_datapath.v \
    $cnn/spectrogram/spectrogram_ctrl.v \
    $cnn/spectrogram/spectrogram.v \
    $cnn/weight_rom.v \
    $cnn/bundle_rom.v \
    $cnn/normalize_datapath.v \
    $cnn/normalize_ctrl.v \
    $cnn/normalize.v \
    $cnn/conv2d_datapath.v \
    $cnn/conv2d_ctrl.v \
    $cnn/conv2d_top.v \
    $cnn/maxpool_datapath.v \
    $cnn/maxpool_ctrl.v \
    $cnn/maxpool.v \
    $cnn/dense_datapath.v \
    $cnn/dense_ctrl.v \
    $cnn/dense.v \
    $cnn/argmax.v \
    $cnn/cnn_core_datapath.v \
    $cnn/cnn_core_ctrl.v \
    $cnn/cnn_core.v \
    $cnn/block_feeder_datapath.v \
    $cnn/block_feeder_ctrl.v \
    $cnn/block_feeder.v \
    $cnn/cnn_seq_ctrl.v \
    $cnn/cnn_path.v \
    $cnn/cnn.v \
]

foreach f [concat $fontes_ml $fontes_cnn] {
    set_global_assignment -name VERILOG_FILE $f
}

# Pinagem, com as posicoes do DE0_CV_golden_top da Terasic.
source $rtl/top/pinout.tcl

export_assignments

execute_flow -compile

# O execute_flow as vezes pula a analise de timing; a chamada explicita
# garante o relatorio de timing.
if {[catch {execute_module -tool sta} erro]} {
    puts "AVISO: analise de timing falhou ($erro)."
    puts "Rode a mao:  quartus_sta $project_name"
}

project_close
