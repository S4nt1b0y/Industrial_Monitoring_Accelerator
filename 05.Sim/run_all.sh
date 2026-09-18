#!/usr/bin/env bash
#
# Runs every testbench under 05.Sim and prints one line per test.
#
#   bash 05.Sim/run_all.sh          # CNN unit tests, ML and top tests, ~1 min
#   bash 05.Sim/run_all.sh --full   # also the 6 real-signal acceptances, ~8 min
#
# Run from Git Bash (iverilog, vvp, make and python3 on PATH). Do not run
# it through WSL's bash: the tools there are different.
#
# Compiled .vvp files and logs go to a temporary folder, printed at the
# end.
# vvp exits 0 even when a test fails, so each log is judged by its text:
# it must say it passed and must not mention a failure or an error.
#
# lms/ and matrix_inv/ only hold empty placeholders and are skipped.

set -u
cd "$(dirname "$0")"

FULL=0
[ "${1:-}" = "--full" ] && FULL=1

OUT="$(mktemp -d)"
RTL=../../04.RTL
ML_SRCS="$RTL/uart/uart_rx.v $RTL/uart/uart_frame_buffer.v \
$RTL/fft/twiddle_lut.v $RTL/fft/addr_gen.v $RTL/fft/butterfly_dif.v \
$RTL/fft/fftu_dif.v $RTL/mdc/mdc.v $RTL/ml_classifier/ml_classifier.v \
$RTL/top/ml_pipeline_fsm.v $RTL/top/ml_pipeline.v"

pass=0
fail=0
failed=""

# run <name> <folder> <command...>
run() {
    local name=$1 dir=$2
    shift 2
    local log="$OUT/$name.log"
    local t0=$SECONDS
    ( cd "$dir" && "$@" ) > "$log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ] \
        && grep -qiE "passed|PASS:" "$log" \
        && ! grep -qiE "fail|error" "$log"; then
        printf "  ok     %-28s %4ss\n" "$name" $((SECONDS - t0))
        pass=$((pass + 1))
    else
        printf "  FALHOU %-28s %4ss  (log: %s)\n" "$name" $((SECONDS - t0)) "$log"
        fail=$((fail + 1))
        failed="$failed $name"
    fi
}

compile_run() {
    local vvp=$1
    shift
    iverilog -o "$OUT/$vvp" "$@" && vvp "$OUT/$vvp"
}

echo "Caminho CNN"
run cnn_unitarios           cnn make all
if [ $FULL -eq 1 ]; then
    for t in tb_cnn tb_cnn_normal tb_cnn_path tb_cnn_path_normal \
             tb_top_wrapper_cnn tb_top_wrapper_cnn_normal; do
        run "$t" cnn make "$t"
    done
fi

echo "Caminho ML e topo"
run tb_uart_frame_buffer    uart          compile_run ufb.vvp -g2005 $RTL/uart/uart_frame_buffer.v tb_uart_frame_buffer.v
run tb_ml_classifier        ml_classifier compile_run mlc.vvp -g2005 $RTL/ml_classifier/ml_classifier.v tb_ml_classifier.v
run tb_mdc                  mdc           compile_run mdc.vvp -g2012 $RTL/mdc/mdc.v tb_mdc.v
run tb_fft                  fft           bash run_fft_compare.sh
run tb_top                  top           compile_run top.vvp -g2005 -s tb_top $ML_SRCS tb_top.v
run tb_top_wrapper          top           compile_run tw.vvp  -g2005 -s tb_top_wrapper \
    $(make -s --no-print-directory -C cnn print-top-srcs) tb_top_wrapper.v

echo
echo "$pass ok, $fail falharam.  Logs em $OUT"
[ $FULL -eq 0 ] && echo "(as 6 aceitacoes com sinal real ficaram de fora; rode com --full)"
[ $fail -eq 0 ] || { echo "Falharam:$failed"; exit 1; }
