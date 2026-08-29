module top_wrapper (
    input  wire clk,
    input  wire rst_n,
    input  wire ml_swith,
    //uart_interface
    output reg Led_Normal,
    output reg Led_Unbalaced,
    output reg Led_disalaighn,
    output reg Led_desgaste
);


//Encher 4 buffers de [DATA_WIDTH-1:0][64]

top #() ();


endmodule