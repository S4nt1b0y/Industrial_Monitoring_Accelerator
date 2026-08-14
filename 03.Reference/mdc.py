def mdc(a, b):
    if a == 0:
        return b

    if b == 0:
        return a

    while a != b:
        if a > b:
            a = a - b
        else:
            b = b - a

    return a


def mdc_tres_picos(pico1, pico2, pico3):
    resultado_parcial = mdc(pico1, pico2)
    resultado_final = mdc(resultado_parcial, pico3)

    return resultado_final


resultado = mdc_tres_picos (12, 18, 30)


# Casos de teste: pico1, pico2, pico3, resultado esperado

testes = [
    (12, 38, 330, 688),
    (8, 1 6, 24, 8),
    (5, 10, 20, 5),
    (0, 18, 30, 6),
    (12, 12, 12, 12)
]

for pico1, pico2, pico3, resultado_esperado in testes:

    resultado_obtido = mdc_tres_picos(pico1, pico2, pico3)

    print("Entradas:", pico1, pico2, pico3)
    print("Resultado esperado:", resultado_esperado)
    print("Resultado obtido:", resultado_obtido)

    if resultado_obtido == resultado_esperado:
        print("Teste aprovado!")
    else:
        print("Teste reprovado!")

    print("--------------------------")
