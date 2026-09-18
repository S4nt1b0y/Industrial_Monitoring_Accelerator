# Clock 50 MHz
set_location_assignment PIN_M9 -to CLOCK_50

# Botão Reset
set_location_assignment PIN_U7 -to KEY[0]

# Switch ML
set_location_assignment PIN_U13 -to SW[0]

# LEDs
set_location_assignment PIN_AA2 -to LEDR[0]
set_location_assignment PIN_AA1 -to LEDR[1]
set_location_assignment PIN_W2  -to LEDR[2]
set_location_assignment PIN_Y3  -to LEDR[3]

# UART RX (CP2102 -> FPGA)
set_location_assignment PIN_N16 -to GPIO_0_0

set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to CLOCK_50
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to KEY[0]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to SW[0]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to GPIO_0_0
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to LEDR[0]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to LEDR[1]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to LEDR[2]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to LEDR[3]

# Portas do top_wrapper sem uso no projeto, posicionadas nos pinos da
# placa para o Quartus não escolher outros. Posições do DE0_CV_golden_top.
set_location_assignment PIN_W9   -to KEY[1]
set_location_assignment PIN_M7   -to KEY[2]
set_location_assignment PIN_M6   -to KEY[3]
set_location_assignment PIN_V13  -to SW[1]
set_location_assignment PIN_T13  -to SW[2]
set_location_assignment PIN_T12  -to SW[3]
set_location_assignment PIN_AA15 -to SW[4]
set_location_assignment PIN_AB15 -to SW[5]
set_location_assignment PIN_AA14 -to SW[6]
set_location_assignment PIN_AA13 -to SW[7]
set_location_assignment PIN_AB13 -to SW[8]
set_location_assignment PIN_AB12 -to SW[9]
set_location_assignment PIN_N2   -to LEDR[4]
set_location_assignment PIN_N1   -to LEDR[5]
set_location_assignment PIN_U2   -to LEDR[6]
set_location_assignment PIN_U1   -to LEDR[7]
set_location_assignment PIN_L2   -to LEDR[8]
set_location_assignment PIN_L1   -to LEDR[9]

set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to KEY[1]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to KEY[2]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to KEY[3]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to SW[1]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to SW[2]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to SW[3]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to SW[4]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to SW[5]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to SW[6]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to SW[7]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to SW[8]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to SW[9]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to LEDR[4]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to LEDR[5]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to LEDR[6]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to LEDR[7]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to LEDR[8]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to LEDR[9]