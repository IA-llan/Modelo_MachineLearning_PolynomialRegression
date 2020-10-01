# Importando as bibliotecas

import pandas as pd
import matplotlib.pyplot as plt

# Importando o dataset e classificando as variaveis em dependentes e independentes

dataset = pd.read_csv('C:/Users/allan/OneDrive/Área de Trabalho/Udemy - Curso/Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Treinando um modelo linear regression em todo o dataset, para que seja feita uma comparacao mais para frente

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Treinando um modelo polynomial Regression

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# Visualizando os resultado da Linear Regression

plt.scatter(x, y, color='yellow')  # Salarios reais
plt.plot(x, lin_reg.predict(x), color='black')  # Predicoes de forma linear
plt.title('Verdade ou bluff(linear regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

# Visualizando os resultado da Polynomial Regression

# x_grid = np.arange(min(x), max(x), 0.1)
# x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color='orange')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color='black')
plt.title('Polynomial regression')
plt.xlabel('LEVEL')
plt.ylabel('Salary')
plt.show()

# Para prever um novo resultado por meio da regressao linear

lin_reg.predict([[6.5]])

# Para prever um novo resultado por meio da regressao polinomial

lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
