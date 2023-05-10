import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Загружаем данные
bank_data = pd.read_csv('data/bank.csv', sep=';')

# Отделяем значения признаков от результата (традиционно y -результат)
bank_features = bank_data.drop('y', axis=1)
bank_output = bank_data.y

# Многие признаки у нас имеют строковые значения и их для регрессии необходимо преобразовать в числа
bank_features = pd.get_dummies(bank_features)
# Результат также переводим в число
bank_output = bank_output.replace({'no': 0, 'yes': 1})
# Разбиваемдатасетначасти-75% дляобучения, 25% -дляпроверки
X_train, X_test, y_train, y_test = train_test_split(bank_features, bank_output, test_size=0.25, random_state=42)

# Создаеммодель
bank_model = LogisticRegression(C=1e6, solver='liblinear')
bank_model.fit(X_train, y_train)
# Рассчитываемполученнуюточность
accuracy_score = bank_model.score(X_train, y_train)
print(accuracy_score)

# Демонстрация проблем с данными -данные не равномерные, что приводит к невысокой точности
plt.bar([0, 1], [len(bank_output[bank_output == 0]), len(bank_output[bank_output == 1])])
plt.xticks([0, 1])
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Выводим относительное количество успешного "впаривания" предложения банка
print('Positive cases: {:.3f}% of all'.format(bank_output.sum() / len(bank_output) * 100))
# На тестовой части проводим прогнозирование
predictions = bank_model.predict(X_test)
# Сверяем прогнозы с данными и выводим отчет
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))
