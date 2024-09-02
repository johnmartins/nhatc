import cexprtk

st = cexprtk.Symbol_Table({}, {}, add_constants=True)
st.variables['a'] = 10

expression = cexprtk.Expression('log10(pi)', st)
print(expression())

q = expression.value()

print(q)


