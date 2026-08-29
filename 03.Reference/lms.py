"""
LMS Filter - Modelo de referência da arquitetura de hardware

Arquitetura:
    - 8 taps
    - 8 amostras armazenadas
    - 8 coeficientes
    - 1 multiplicador compartilhado
    - 1 acumulador
    - cálculo sequencial de y(n)
    - cálculo do erro e(n)
    - cálculo de mu * e(n)
    - atualização sequencial dos coeficientes

Equações:

    y(n) = sum(w[i] * x[n-i])

    e(n) = d(n) - y(n)

    delta_w[i] = mu * e(n) * x[n-i]

    w[i] = w[i] + delta_w[i]

Observação:
    Este programa é um modelo de referência em ponto flutuante.
    O HDL poderá posteriormente usar ponto fixo.
"""


class LMSHardwareModel:
    def __init__(self, num_taps=8, mu=0.01):
        """
        Inicializa a arquitetura do LMS.

        num_taps:
            quantidade de coeficientes/taps.

        mu:
            passo de adaptação.
        """

        # Número de taps do filtro.
        self.num_taps = num_taps

        # Passo de adaptação.
        self.mu = mu

        # -------------------------------------------------
        # BANCO DE AMOSTRAS
        #
        # x_buffer[0] = x(n)
        # x_buffer[1] = x(n-1)
        # ...
        # x_buffer[7] = x(n-7)
        # -------------------------------------------------
        self.x_buffer = [0.0] * num_taps

        # -------------------------------------------------
        # BANCO DE COEFICIENTES
        #
        # w[0], w[1], ..., w[7]
        # -------------------------------------------------
        self.w = [0.0] * num_taps

        # Resultado da saída do filtro.
        self.y = 0.0

        # Erro.
        self.error = 0.0

        # Resultado intermediário:
        # mu * error
        self.mu_error = 0.0

        # Acumulador.
        self.accumulator = 0.0

        # Índice do tap atualmente processado.
        self.tap = 0

    # =====================================================
    # LOAD
    # =====================================================

    def load_sample(self, x_new, d_new):
        """
        Simula o estado LOAD do hardware.

        x_new:
            nova amostra do sinal de referência.

        d_new:
            nova amostra do sinal desejado/principal.
        """

        # Desloca o histórico de amostras.
        #
        # x[7] <- x[6]
        # x[6] <- x[5]
        # ...
        # x[1] <- x[0]
        #
        # Em Python fazemos isso do final para o começo.
        for i in range(self.num_taps - 1, 0, -1):
            self.x_buffer[i] = self.x_buffer[i - 1]

        # Coloca a nova amostra na posição 0.
        self.x_buffer[0] = x_new

        # Guarda o sinal desejado da amostra atual.
        self.d = d_new

        # Zera o acumulador para iniciar o cálculo de y(n).
        self.accumulator = 0.0

        # Começa pelo tap 0.
        self.tap = 0

    # =====================================================
    # CALC_Y
    # =====================================================

    def calculate_y(self):
        """
        Simula CALC_Y usando UM ÚNICO multiplicador.

        Em cada iteração:

            produto = x[tap] * w[tap]

        Depois:

            accumulator = accumulator + produto
        """

        # Calcula sequencialmente os 8 produtos.
        for i in range(self.num_taps):

            # -------------------------------------------------
            # UM ÚNICO MULTIPLICADOR
            # -------------------------------------------------
            product = self.x_buffer[i] * self.w[i]

            # Acumula o produto.
            self.accumulator += product

        # Depois dos 8 produtos:
        #
        # accumulator = y(n)
        self.y = self.accumulator

        return self.y

    # =====================================================
    # CALC_E
    # =====================================================

    def calculate_error(self):
        """
        Simula o estado CALC_E.

            e(n) = d(n) - y(n)
        """

        self.error = self.d - self.y

        return self.error

    # =====================================================
    # CALC_MU_E
    # =====================================================

    def calculate_mu_error(self):
        """
        Simula o cálculo:

            mu_error = mu * error
        """

        self.mu_error = self.mu * self.error

        return self.mu_error

    # =====================================================
    # UPDATE
    # =====================================================

    def update_weights(self):
        """
        Simula UPDATE.

        Para cada tap:

            delta_w = mu * error * x[i]

            w[i] = w[i] + delta_w

        O hardware usa novamente o único multiplicador
        compartilhado.
        """

        for i in range(self.num_taps):

            # -------------------------------------------------
            # SEGUNDA UTILIZAÇÃO DO MULTIPLICADOR
            #
            # mu_error * x[i]
            # -------------------------------------------------
            delta_w = self.mu_error * self.x_buffer[i]

            # Atualiza o coeficiente.
            self.w[i] += delta_w

    # =====================================================
    # PROCESS_SAMPLE
    # =====================================================

    def process_sample(self, x_new, d_new):
        """
        Executa uma iteração completa do LMS.

        Sequência equivalente à FSM:

            LOAD
              ↓
            CALC_Y
              ↓
            CALC_E
              ↓
            CALC_MU_E
              ↓
            UPDATE
              ↓
            DONE
        """

        # -------------------------
        # LOAD
        # -------------------------
        self.load_sample(x_new, d_new)

        # -------------------------
        # CALC_Y
        # -------------------------
        y = self.calculate_y()

        # -------------------------
        # CALC_E
        # -------------------------
        error = self.calculate_error()

        # -------------------------
        # CALC_MU_E
        # -------------------------
        mu_error = self.calculate_mu_error()

        # -------------------------
        # UPDATE
        # -------------------------
        self.update_weights()

        # Retorna os principais resultados.
        return {
            "y": y,
            "error": error,
            "mu_error": mu_error,
            "weights": self.w.copy(),
        }


# =========================================================
# EXEMPLO DE UTILIZAÇÃO
# =========================================================

if __name__ == "__main__":

    # Cria o LMS com:
    #
    # 8 taps
    # mu = 0.01
    #
    lms = LMSHardwareModel(
        num_taps=8,
        mu=0.01
    )

    # Exemplos de sinais.
    #
    # Aqui estamos usando números arbitrários apenas
    # para demonstrar o funcionamento.
    x_signal = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0
    ]

    d_signal = [
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0
    ]

    # Processa uma amostra por vez.
    for n in range(len(x_signal)):

        result = lms.process_sample(
            x_new=x_signal[n],
            d_new=d_signal[n]
        )

        print("=" * 70)
        print(f"Amostra n = {n}")

        print(f"x(n)       = {x_signal[n]:.6f}")
        print(f"d(n)       = {d_signal[n]:.6f}")
        print(f"y(n)       = {result['y']:.6f}")
        print(f"e(n)       = {result['error']:.6f}")
        print(f"mu * e(n)  = {result['mu_error']:.6f}")

        print("Pesos:")
        for i, weight in enumerate(result["weights"]):
            print(f"  w[{i}] = {weight:.8f}")
