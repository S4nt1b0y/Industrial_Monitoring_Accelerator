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