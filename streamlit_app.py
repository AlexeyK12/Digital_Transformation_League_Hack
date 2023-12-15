import pandas as pd 
import numpy as np
import time
import io
import os
import streamlit as st
import psycopg2
from sqlalchemy import create_engine
import openpyxl
from IPython.display import display, HTML, Javascript, Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from catboost import CatBoostClassifier
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
random.seed(42)



# заголовок
st.markdown("<center><h1>Контроль эффективности сотрудников</h1></center>", unsafe_allow_html=True)

# прогресс-бар
def long_running_process():
    for i in range(100):
        time.sleep(0.03)  
        progress_percent = (i + 1) / 100.0
        progress_bar.progress(progress_percent)

#st.markdown('**соединение с базой данных и прогнозирование**')
progress_bar = st.progress(0)
long_running_process()
st.markdown(
    """<style>
       div[data-baseweb="progress-bar"] {
           display: none;
       }
     </style>""",
    unsafe_allow_html=True
)

# увеличиваем лимит скорости для БД
display(Javascript('''
    var kernel = IPython.notebook.kernel;
    kernel.execute("from notebook.services.config import ConfigManager");
    kernel.execute("cm = ConfigManager().update('NotebookApp', {'iopub_data_rate_limit': 10000000})");
    '''))

# подключение к БД Postgre
db_params = {'host': '46.243.226.196',
             'database': 'lct',
             'user': 'postgres',
             'password': '2S7sbzhjAw7',
             'port': '5432'}

connection = psycopg2.connect(**db_params)
cursor = connection.cursor()

try:
    connection = psycopg2.connect(**db_params)
    print("Соединение с базой данных установлено")
except (Exception, psycopg2.Error) as error:
    print("Ошибка при подключении к базе данных:", error)

# выгружаем прогнозы из БД 
sql_query = """
    SELECT *
    FROM public.dataset_V
            """
cursor.execute(sql_query)

# формируем датафрейм из запроса БД
rows = cursor.fetchall()

data_to_insert = []
for row in rows:
    data_to_insert.append(row)

df = pd.DataFrame(data_to_insert, columns=['id', 'фио', 'date', 'period', 'месячный_доход',
                                            'получаемых_сообщений_за_период',
                                            'отправленных_сообщений_за_период',
                                            'адресатов_в_отправляемых_сообщениях',
                                            'сообщений_с_адресами_в_скрытой_копии',
                                            'сообщений_с_адресатами_в_поле_копия',
                                            'сообщений._прочитанных_после_4_часов', 'дней_между_получено_прочитано',
                                            'ответо_на_сообщения', 'символов_в_исходящих_сообщениях',
                                            'сообщений._отправленных_вне_рамок_рабочего_дня',
                                            'соотношение_полученных_отправленных',
                                            'соотношение_объема_полученных_отправленных_байт',
                                            'входящих_сообщений_с_._без_ответа', 'target'])

# закрываем соединение с БД
cursor.close()
connection.close()

# генерируем календарный признаки, удаляем выходные + период
df['date'] = pd.to_datetime(df['date'])
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['weekday'] = df['date'].dt.weekday + 1
df['is_weekend'] = df['weekday'].isin([6, 7]).astype(int)
df = df.query('is_weekend != 1')
df.drop(columns='is_weekend', inplace=True)
df['period'] = (df['date'] - df['date'].min()).dt.days // 7 + 1
df = df.sort_values('date')

exclude_columns = ['соотношение_полученных_отправленных',
                   'соотношение_объема_полученных_отправленных_байт',
                   'id', 'date', 'фио']

for column in df.columns:
    if column not in exclude_columns:
        df[column] = df[column].astype(int)

st.write(df)

# кнопка скачивания данных
export_button = st.button("Скачать данные в CSV")

# условие выгрузки
if export_button:
    app_directory = os.path.dirname(os.path.abspath(__file__))
    csv_filename = "контроль_эффективности_база.csv"
    csv_filepath = os.path.join(app_directory, csv_filename)
    df.to_csv(csv_filepath, index=False)
    st.success(f"Данные выгружены в {csv_filepath}")

# дашборд в Grafana
def open_dashboard():
    dashboard_url = "http://46.243.226.196:3000/d/cb0b997a-68dc-46cc-809f-18bbbd4c666d/verojatnost--uvol-nenija?orgId=1"
    st.markdown(
        f'<a href="{dashboard_url}" target="_blank" style="color: white; background-color: orange; padding: 20px 30px; border-radius: 35px; text-decoration: none; display: inline-block; text-align: center; font-size: 24px;">Аналитика в Grafana</a>',
        unsafe_allow_html=True
    )
open_dashboard()

# выбор периода
st.sidebar.markdown("""
<style>
    .sidebar-widget {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .sidebar-widget label {
        font-weight: bold;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown(
    '<div class="sidebar-widget" style="color: black;">Выберите период</div>',
    unsafe_allow_html=True
)
max_period = df['period'].max()
selected_period = st.sidebar.number_input("", min_value=df['period'].min(), max_value=df['period'].max(), value=max_period, step=1, key="selected_period")

# формирование обучающей и контрольной выборок
train = df.query(f'period < {selected_period}')
test = df.query(f'period == {selected_period}')
X_train = train.drop(columns=['date', 'target'])
y_train = train['target']
y_test = test['target']
X_test = test.drop(columns=['date', 'target'])

# обучение CatBoost
model = CatBoostClassifier(cat_features=['id', 'фио'], 
                           verbose=0,
                           random_seed=42,
                           n_estimators=100)

model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# установка порога
y_pred = (y_pred_proba > 0.05).astype(int)

# общий результат по организации
def color_probas(val):
    if val < 0.25:
        return 'green'
    else:
        return 'red'

X_test['вероятность_увольнения'] = y_pred_proba
result = (X_test.groupby(['id', 'фио'], as_index=False).
          agg({'вероятность_увольнения':'mean'}).
          sort_values('вероятность_увольнения', ascending=False))

result['color'] = result['вероятность_увольнения'].apply(color_probas)
result_styled = result.copy()

# повторное подключение к БД Postgres
db_params = {
    'host': '46.243.226.196',
    'database': 'lct',
    'user': 'postgres',
    'password': '2S7sbzhjAw7',
    'port': '5432'
}

try:
    connection = psycopg2.connect(**db_params)
    print("Соединение с базой данных установлено")
except (Exception, psycopg2.Error) as error:
    print("Ошибка при подключении к базе данных:", error)

# сохранем результаты на период
result_df = pd.DataFrame(result, columns=['id', 'фио', 'вероятность_увольнения'])
engine = create_engine(
    f'postgresql://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/{db_params["database"]}'
) 
result_df.to_sql('results', engine, index=False, if_exists='replace')  

# основной дашборд
fig = px.bar(result.sort_values('вероятность_увольнения', ascending=False).
             query('вероятность_увольнения > 0.05'),
             x='id', y='вероятность_увольнения', text='фио',
             color='color', color_discrete_sequence=['red', 'lightseagreen'],
             labels={'color': 'Категория'})  
fig.update_traces(texttemplate='%{text}', textposition='inside', width=0.8)
fig.update_layout(
    title='Вероятности увольнения по организации',
    xaxis=dict(title='ID'),
    yaxis=dict(title='вероятность'),
    barmode='group',
    bargap=0.15,
    uniformtext_minsize=12,
    margin=dict(b=100),
)

# подпись к легенде
fig.update_layout(
    legend=dict(
        title='Категории',
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1
    )
)

fig.for_each_trace(lambda t: t.update(name='Подозрительно') if t.name == 'red' else t.update(name='Нейтрально'))
st.plotly_chart(fig)


# отображаем таблицу с условным форматированием
result_styled_subset = result_styled[['id', 'фио', 'вероятность_увольнения']]

def color_probas(val):
    if val < 0.25:
        return 'background-color: lightseagreen'
    else:
        return 'background-color: red'

result_styled_subset = result_styled_subset.style.applymap(color_probas, subset=['вероятность_увольнения'])
st.dataframe(result_styled_subset, height=600)

output_excel = io.BytesIO()
result_styled[['id', 'фио', 'вероятность_увольнения']].to_excel(output_excel, index=False)
output_excel.seek(0)  

download_button = st.download_button(
    label="Скачать данные за период",
    data=output_excel,
    file_name=f"данные_за_период_{selected_period}.xlsx",
    key="downloadButton"
)

result_styled_subset = result_styled[['id', 'фио', 'вероятность_увольнения']]

def color_probas(val):
    if val < 0.25:
        return 'background-color: green'
    else:
        return 'background-color: red'

# результаты по отделам
it = result[result['id'].str.startswith('it')].sort_values('вероятность_увольнения', ascending=False)
it_result = it[['id', 'фио', 'вероятность_увольнения']].style.applymap(color_probas, subset=['вероятность_увольнения'])
administration = result[result['id'].str.startswith('adm')].sort_values('вероятность_увольнения', ascending=False)
administration_result = administration[['id', 'фио', 'вероятность_увольнения']].style.applymap(color_probas, subset=['вероятность_увольнения'])
finance = result[result['id'].str.startswith('fin')].sort_values('вероятность_увольнения', ascending=False)
finance_result = finance[['id', 'фио', 'вероятность_увольнения']].style.applymap(color_probas, subset=['вероятность_увольнения'])
production = result[result['id'].str.startswith('prod')].sort_values('вероятность_увольнения', ascending=False)
production_result = production[['id', 'фио', 'вероятность_увольнения']].style.applymap(color_probas, subset=['вероятность_увольнения'])
kommercian = result[result['id'].str.startswith('kom')].sort_values('вероятность_увольнения', ascending=False)
kommercian_result = kommercian[['id', 'фио', 'вероятность_увольнения']].style.applymap(color_probas, subset=['вероятность_увольнения'])


# отображение дашбордов и таблиц по отделам
def show_dashboard_and_table(department_df, department_name):
    department_result = department_df[['id', 'фио', 'вероятность_увольнения']].style.applymap(color_probas, subset=['вероятность_увольнения'])
    
    # условие для отображения и скачивания данных
    if st.sidebar.button(f'{department_name}'):
        with st.container():
            st.subheader(department_name)
            fig = px.bar(department_df.query('вероятность_увольнения > 0.05'),
                         x='id', y='вероятность_увольнения', color='color', text='фио',
                         color_discrete_sequence=['red', 'green'])  
            fig.update_traces(texttemplate='%{text}', textposition='inside', width=0.8)
            fig.update_layout(
                title=f'Вероятности увольнения - {department_name}',
                xaxis=dict(title='ID'),
                yaxis=dict(title='вероятность'),
                barmode='group',
                bargap=0.15,
                uniformtext_minsize=12,
                margin=dict(b=100)
            )
            
            # легенда
            fig.update_layout(
                legend=dict(
                    title='Категории',
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                )
            )
            
            # подписи к легенде
            fig.for_each_trace(lambda t: t.update(name='Подозрительно') if t.name == 'red' else t.update(name='Нейтрально'))

            st.plotly_chart(fig)
            st.dataframe(department_result)

            # скачивание данных в excel
            excel_filename = f"{department_name}_период_{selected_period}.xlsx"
            app_directory = os.path.dirname(os.path.abspath(__file__))
            excel_filepath = os.path.join(app_directory, excel_filename)
            department_df[['id', 'фио', 'вероятность_увольнения']].to_excel(excel_filepath, index=False)
            st.success(f"Данные скачаны в {excel_filepath}")

st.sidebar.title("Выбор подразделения")

            
show_dashboard_and_table(administration, "Администрация")
show_dashboard_and_table(kommercian, "Коммерческий отдел")
show_dashboard_and_table(production, "Производство")
show_dashboard_and_table(finance, "Финансовый отдел")
show_dashboard_and_table(it, "Отдел информационных технологий")


# модуль рассылки по подразделениям
## до ввода реальных адресов установлена заглушка о успешной рассылке
def send_email(subject, body, to_email):
    # почтовые настройки
    smtp_server = 'smtp.yourmailserver.com'
    smtp_port = 587
    smtp_username = 'your_username'
    smtp_password = 'your_password'

    # формирование письма
    message = MIMEMultipart()
    message['From'] = smtp_username
    message['To'] = to_email
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))

    # подключение к серверу и отправка письма
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.sendmail(smtp_username, to_email, message.as_string())

send_email_button = st.button("Отправить рассылку начальникам отделов")

if send_email_button:
    chiefs_df = pd.DataFrame({
        'Отдел': ['it', 'adm', 'fin', 'prod', 'kom'],
        'Email': ['it_chief@example.com', 'adm_chief@example.com', 'fin_chief@example.com', 'prod_chief@example.com', 'kom_chief@example.com']
    })
    for index, row in chiefs_df.iterrows():
        department_name = row['Отдел']
        chief_email = row['Email']
        email_subject = f"Контроль качества работы подразделения {department_name}"
        email_body = f"Начальнику подразделения - {department_name},\n\n"
        department_df = result[result['id'].str.startswith(department_name)].sort_values('вероятность_увольнения', ascending=False)
        email_body += department_df.to_string(index=False)
        #send_email(email_subject, email_body, chief_email)
        #st.warning(f"Не найден адрес электронной почты для начальника отдела {department_name}")

    st.success("Рассылка успешно отправлена.")


# закрываем соединение с БД
cursor.close()
connection.close()
