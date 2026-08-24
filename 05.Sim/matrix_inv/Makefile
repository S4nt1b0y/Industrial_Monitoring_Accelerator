IVERILOG ?= iverilog
VVP      ?= vvp

RTL_COMMON := ../../04.RTL/matrix_inv/common
RTL_DIR    := ../../04.RTL/matrix_inv

IFLAGS := -g2005 -I$(RTL_COMMON)

.PHONY: help all tb_divider tb_gj tb_cof clean

help:
	@echo "Alvos disponiveis:"
	@echo "  tb_divider   Regressao do divisor Q8.8 isolado (8 casos)"
	@echo "  tb_gj        Testbench Gauss-Jordan, N=2/3/4 (36 casos) -- validado"
	@echo "  tb_cof       Testbench Cofatores, N=2/3/4 (36 casos) -- ainda em validacao"
	@echo "  all          tb_divider + tb_gj (o que ja esta validado)"
	@echo "  clean        Remove os .vvp compilados"

tb_divider:
	$(IVERILOG) $(IFLAGS) -o common/tb_divider_q88.vvp \
		$(RTL_COMMON)/divider_q88.v \
		common/tb_divider_q88.v
	$(VVP) common/tb_divider_q88.vvp

tb_gj:
	$(IVERILOG) $(IFLAGS) -o tb_matrix_inv_gj.vvp \
		$(RTL_COMMON)/divider_q88.v \
		$(RTL_DIR)/matrix_inv_gj.v \
		tb_matrix_inv_gj.v
	$(VVP) tb_matrix_inv_gj.vvp

tb_cof:
	$(IVERILOG) $(IFLAGS) -o tb_matrix_inv_cof.vvp \
		$(RTL_COMMON)/divider_q88.v \
		$(RTL_COMMON)/det_q88.v \
		$(RTL_DIR)/matrix_inv_cof.v \
		tb_matrix_inv_cof.v
	$(VVP) tb_matrix_inv_cof.vvp

all: tb_divider tb_gj

clean:
	rm -f tb_matrix_inv_gj.vvp tb_matrix_inv_cof.vvp common/tb_divider_q88.vvp
