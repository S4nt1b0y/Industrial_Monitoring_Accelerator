# matrix_inv

Four algorithm variants, one file each, all sharing the same external
interface so `05.Sim/matrix_inv` can test them the same way.

| Module               | Algorithm                      | Status                  |
| -------------------- | ------------------------------ | ----------------------- |
| `matrix_inv_gj.v`  | Gauss-Jordan, partial pivoting | N=2/3/4 passing (36/36) |
| `matrix_inv_cof.v` | Cofactors / adjugate           | not implemented         |
| `matrix_inv_lu.v`  | LU decomposition               | not implemented         |
| `matrix_inv_qr.v`  | QR decomposition (Givens)      | not implemented         |

`common/` holds shared building blocks: `divider_q88.v` (iterative
shift-subtract divider, reused by both `matrix_inv_gj` and
`matrix_inv_cof`, the latter at a wider `WORD_BITS` — see below),
`det_q88.v` (recursive determinant, `matrix_inv_cof` only), and
`q88_ops.vh` (shared saturating scalar functions, `included).

## Common interface

```
module matrix_inv_<algo> #(
    parameter N         = 4,   // 2, 3 or 4
    parameter WORD_BITS = 16,  // Q8.8
    parameter FRAC_BITS = 8
) (
    input  wire                            clk,
    input  wire                            rst_n,

    input  wire                            start,
    input  wire signed [N*N*WORD_BITS-1:0] a_in,     // row-major, flattened
    input  wire signed [N*WORD_BITS-1:0]   b_in,     // flattened

    output reg                             busy,
    output reg                             done,     // 1-cycle pulse
    output reg                             singular, // pivot < epsilon
    output reg  signed [N*WORD_BITS-1:0]   x_out,    // flattened
    output reg  signed [N*N*WORD_BITS-1:0] ainv_out  // row-major, flattened
);
```

- Ports are flattened buses, not unpacked array ports. Element `(i, j)`
  of `a_in` lives at bits `[(i*N+j+1)*WORD_BITS-1 -: WORD_BITS]`; element
  `i` of `b_in`/`x_out` at `[(i+1)*WORD_BITS-1 -: WORD_BITS]`. Each
  algorithm module unpacks into a local `reg signed [WORD_BITS-1:0] mem [0:N*N-1]` array internally.
- `a_in`/`b_in` are captured on the `start` pulse; results are valid for
  one cycle on the `done` pulse (registered outputs, so they hold until
  the next `start`).
- `singular` follows `EPSILON_Q88 = 4` (raw Q8.8 units) . When asserted, `x_out`/`ainv_out`
  content is undefined for that run.
- Internal datapath width is left to each algorithm's implementation, not
  part of the port list — see the Sprint 0 fixed-point note (32-bit/Q16.16
  MAC intermediates, rescaled to Q8.8 on write-back). `matrix_inv_cof`
  additionally uses a 32-bit `WIDE_BITS` container for its internal
  cofactor/subdeterminant signals — plain Q8.8 overflows there even for
  well-conditioned matrices.
