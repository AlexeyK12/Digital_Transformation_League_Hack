![Image](fon_rep.png)


# <a style="color:cornflowerblue;">Разработка комплексного аналитического решения для контроля эффективности сотрудников, включая визуализацию, машинное обучение и взаимодействие с базой данных</a>
![Python](https://sun9-73.userapi.com/impg/mBPiNRTUpoGbvGmZjhd6wwyA9lufh7OpV5aOZA/9glj_FxN88E.jpg?size=604x381&quality=96&sign=713e7202a77e0bc8db39aad05282b8ca&type=album)

![Streamlit](https://images.datacamp.com/image/upload/v1640050215/image27_frqkzv.png)
![Plotly](https://images.plot.ly/plotly-documentation/images/logo/new-branding/plotly-logomark.png)
![CatBoost](https://catboost.ai/docs/logo/catboost-logo.png)
![Docker](https://www.docker.com/sites/default/files/d8/2019-07/vertical-logo-monochromatic.png)
![Git](https://git-scm.com/images/logos/downloads/Git-Logo-2Color.png)

### Модуль 1: Подключение к Базе Данных
Задание параметров подключения к базе данных PostgreSQL.
Использование библиотеки psycopg2 для установления соединения.
Выполнение SQL-запроса для извлечения данных о сотрудниках.

### Модуль 2: Обработка и Подготовка Данных
Преобразование дат в удобный формат и создание календарных признаков.
Преобразование данных и подготовка DataFrame с использованием библиотеки pandas.

### Модуль 3: Визуализация и Анализ Данных
Вывод и исследование структуры данных БД.
Создание дашборда с вероятностью увольнения для каждого сотрудника за выбранный период.
Использование библиотек Plotly и Matplotlib для создания графиков.

### Модуль 4: Машинное Обучение и Прогнозирование
Формирование обучающей и контрольной выборок для обучения модели машинного обучения.
Использование библиотеки CatBoostClassifier, работающей с задачами бинарной классификации, для обучения модели вероятности увольнения сотрудника.
Предусмотрена возможность работы с категориальными признаками, в том числе добавление новых фичей.
Установка порога для классификации сотрудников по вероятности увольнения.

### Модуль 5: Вывод Результатов и Экспорт Данных
Визуализация результатов в виде интерактивных графиков и таблиц с цветовой кодировкой.
Возможность скачивания данных в формате CSV и Excel для последующего анализа.
Предоставление доступа к дашборду в Grafana для более детального изучения данных.

### Модуль 6: Отправка Результатов и Рассылка
Организация отправки электронных писем с результатами анализа начальникам отделов.
Временная заглушка о успешной рассылке, пока не будут предоставлены реальные адреса электронной почты.

### Модуль 7: Завершение Работы и Закрытие Соединения
Закрытие соединения с базой данных PostgreSQL после завершения всех операций.
Завершение работы веб-приложения с использованием Streamlit.

### Данное приложение является инструментом для анализа и прогнозирования, позволяющим принимать обоснованные управленческие решения.

## **Инструменты проекта:**

1. **Язык программирования:**
   - Python

2. **Библиотеки для обработки данных:**
   - Pandas
   - NumPy

3. **Визуализация данных:**
   - Matplotlib
   - Plotly

4. **Веб-приложение:**
   - Streamlit

5. **Взаимодействие с базой данных:**
   - Psycopg2 (для работы с PostgreSQL)
   - SQLAlchemy

6. **Машинное обучение:**
   - CatBoostClassifier

7. **Работа с электронной почтой:**
    - MIMEText, MIMEMultipart
    - smtplib

8. **Генерация случайных данных:**
   - Random (для управления случайным зерном)

9. **Веб-документация:**
   - Markdown

10. **Дополнительные инструменты для визуализации при разработке:**
    - IPython.display
    - HTML, Javascript

11. **Интерактивные графики и дашборды:**
    - Grafana (для визуализации внешних дашбордов)

12. **Системные операции с файлами и директориями:**
    - io
    - os
