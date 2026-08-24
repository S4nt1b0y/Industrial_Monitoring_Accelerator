function signed [WORD_BITS-1:0] q88_mul;
    input signed [WORD_BITS-1:0] a;
    input signed [WORD_BITS-1:0] b;
    reg signed [2*WORD_BITS-1:0] prod;
    reg signed [2*WORD_BITS-1:0] rounded;
    begin
        prod = a * b;
        rounded = (prod + (1 <<< (FRAC_BITS-1))) >>> FRAC_BITS;
        if (rounded > $signed({1'b0, {(WORD_BITS-1){1'b1}}}))
            q88_mul = {1'b0, {(WORD_BITS-1){1'b1}}};
        else if (rounded < $signed({1'b1, {(WORD_BITS-1){1'b0}}}))
            q88_mul = {1'b1, {(WORD_BITS-1){1'b0}}};
        else
            q88_mul = rounded[WORD_BITS-1:0];
    end
endfunction

function signed [WORD_BITS-1:0] q88_add_sat;
    input signed [WORD_BITS-1:0] a;
    input signed [WORD_BITS-1:0] b;
    reg signed [WORD_BITS:0] sum;
    begin
        sum = a + b;
        if (sum > $signed({1'b0, {(WORD_BITS-1){1'b1}}}))
            q88_add_sat = {1'b0, {(WORD_BITS-1){1'b1}}};
        else if (sum < $signed({1'b1, {(WORD_BITS-1){1'b0}}}))
            q88_add_sat = {1'b1, {(WORD_BITS-1){1'b0}}};
        else
            q88_add_sat = sum[WORD_BITS-1:0];
    end
endfunction

function signed [WIDE_BITS-1:0] q88_mul_wide;
    input signed [WORD_BITS-1:0] a;      // plain Q8.8 element
    input signed [WIDE_BITS-1:0] b;      // wide operand
    reg signed [WORD_BITS+WIDE_BITS-1:0] prod;
    reg signed [WORD_BITS+WIDE_BITS-1:0] rounded;
    begin
        prod = a * b;
        rounded = (prod + (1 <<< (FRAC_BITS-1))) >>> FRAC_BITS;
        if (rounded > $signed({1'b0, {(WIDE_BITS-1){1'b1}}}))
            q88_mul_wide = {1'b0, {(WIDE_BITS-1){1'b1}}};
        else if (rounded < $signed({1'b1, {(WIDE_BITS-1){1'b0}}}))
            q88_mul_wide = {1'b1, {(WIDE_BITS-1){1'b0}}};
        else
            q88_mul_wide = rounded[WIDE_BITS-1:0];
    end
endfunction

function signed [WIDE_BITS-1:0] q88_add_sat_wide;
    input signed [WIDE_BITS-1:0] a;
    input signed [WIDE_BITS-1:0] b;
    reg signed [WIDE_BITS:0] sum;
    begin
        sum = a + b;
        if (sum > $signed({1'b0, {(WIDE_BITS-1){1'b1}}}))
            q88_add_sat_wide = {1'b0, {(WIDE_BITS-1){1'b1}}};
        else if (sum < $signed({1'b1, {(WIDE_BITS-1){1'b0}}}))
            q88_add_sat_wide = {1'b1, {(WIDE_BITS-1){1'b0}}};
        else
            q88_add_sat_wide = sum[WIDE_BITS-1:0];
    end
endfunction

function signed [WIDE_BITS-1:0] q88_neg_sat_wide;
    input signed [WIDE_BITS-1:0] a;
    begin
        if (a == $signed({1'b1, {(WIDE_BITS-1){1'b0}}})) // negating WIDE_MIN overflows
            q88_neg_sat_wide = {1'b0, {(WIDE_BITS-1){1'b1}}};
        else
            q88_neg_sat_wide = -a;
    end
endfunction

function signed [WORD_BITS-1:0] q88_saturate_wide_to_word;
    input signed [WIDE_BITS-1:0] a;
    begin
        if (a > $signed({{(WIDE_BITS-WORD_BITS){1'b0}}, 1'b0, {(WORD_BITS-1){1'b1}}}))
            q88_saturate_wide_to_word = {1'b0, {(WORD_BITS-1){1'b1}}};
        else if (a < $signed({{(WIDE_BITS-WORD_BITS){1'b1}}, 1'b1, {(WORD_BITS-1){1'b0}}}))
            q88_saturate_wide_to_word = {1'b1, {(WORD_BITS-1){1'b0}}};
        else
            q88_saturate_wide_to_word = a[WORD_BITS-1:0];
    end
endfunction

function signed [WIDE_BITS-1:0] q88_sign_extend_to_wide;
    input signed [WORD_BITS-1:0] a;
    begin
        q88_sign_extend_to_wide = {{(WIDE_BITS-WORD_BITS){a[WORD_BITS-1]}}, a};
    end
endfunction
