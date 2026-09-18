# CLOCK_50: oscilador de 50 MHz da DE0-CV, clock definido no enunciado.
# Sem create_clock o Timing Analyzer usa um periodo padrao de 1 ns.
create_clock -name CLOCK_50 -period 20.000 [get_ports {CLOCK_50}]
derive_clock_uncertainty

# GPIO_0_0 (TX do CP2102) e assincrono ao CLOCK_50; o uart_rx sincroniza
# a entrada. Nao ha restricao externa de setup/hold.
set_false_path -from [get_ports {GPIO_0_0}]

# KEY e SW sao entradas mecanicas, sem relacao temporal com o clock.
set_false_path -from [get_ports {KEY[*]}]
set_false_path -from [get_ports {SW[*]}]

# Os LEDs nao sao amostrados por nenhum circuito externo.
set_false_path -to [get_ports {LEDR[*]}]
