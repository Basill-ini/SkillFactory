#!/usr/bin/env python
# coding: utf-8

# ### Задача:
# Вас пригласили поучаствовать в одном из проектов UNICEF — международного подразделения ООН, чья миссия состоит в повышении уровня благополучия детей по всему миру. 
# 
# Суть проекта — отследить влияние условий жизни учащихся в возрасте от 15 до 22 лет на их успеваемость по математике, чтобы на ранней стадии выявлять студентов, находящихся в группе риска.
# 
# И сделать это можно с помощью модели, которая предсказывала бы результаты госэкзамена по математике для каждого ученика школы. Чтобы определиться с параметрами будущей модели, проведите разведывательный анализ данных и составьте отчёт по его результатам. 

# ### Описание датасета:

# 1. school — аббревиатура школы, в которой учится ученик
# 
# 2. sex — пол ученика ('F' - женский, 'M' - мужской)
# 
# 3. age — возраст ученика (от 15 до 22)
# 
# 4. address — тип адреса ученика ('U' - городской, 'R' - за городом)
# 
# 5. famsize — размер семьи('LE3' <= 3, 'GT3' >3)
# 
# 6. Pstatus — статус совместного жилья родителей ('T' - живут вместе 'A' - раздельно)
# 
# 7. Medu — образование матери (0 - нет, 1 - 4 класса, 2 - 5-9 классы, 3 - среднее специальное или 11 классов, 4 - высшее)
# 
# 8. Fedu — образование отца (0 - нет, 1 - 4 класса, 2 - 5-9 классы, 3 - среднее специальное или 11 классов, 4 - высшее)
# 
# 9. Mjob — работа матери ('teacher' - учитель, 'health' - сфера здравоохранения, 'services' - гос служба, 'at_home' - не работает, 'other' - другое)
# 
# 10. Fjob — работа отца ('teacher' - учитель, 'health' - сфера здравоохранения, 'services' - гос служба, 'at_home' - не работает, 'other' - другое)
# 
# 11. reason — причина выбора школы ('home' - близость к дому, 'reputation' - репутация школы, 'course' - образовательная программа, 'other' - другое)
# 
# 12. guardian — опекун ('mother' - мать, 'father' - отец, 'other' - другое)
# 
# 13. traveltime — время в пути до школы (1 - <15 мин., 2 - 15-30 мин., 3 - 30-60 мин., 4 - >60 мин.)
# 
# 14. studytime — время на учёбу помимо школы в неделю (1 - <2 часов, 2 - 2-5 часов, 3 - 5-10 часов, 4 - >10 часов)
# 
# 15. failures — количество внеучебных неудач (n, если 1<=n<3, иначе 0)
# 
# 16. schoolsup — дополнительная образовательная поддержка (yes или no)
# 
# 17. famsup — семейная образовательная поддержка (yes или no)
# 
# 18. paid — дополнительные платные занятия по математике (yes или no)
# 
# 19. activities — дополнительные внеучебные занятия (yes или no)
# 
# 20. nursery — посещал детский сад (yes или no)
# 
# 21. higher — хочет получить высшее образование (yes или no)
# 
# 22. internet — наличие интернета дома (yes или no)
# 
# 23. romantic — в романтических отношениях (yes или no)
# 
# 24. famrel — семейные отношения (от 1 - очень плохо до 5 - очень хорошо)
# 
# 25. freetime — свободное время после школы (от 1 - очень мало до 5 - очень мого)
# 
# 26. goout — проведение времени с друзьями (от 1 - очень мало до 5 - очень много)
# 
# 27. health — текущее состояние здоровья (от 1 - очень плохо до 5 - очень хорошо)
# 
# 28. absences — количество пропущенных занятий
# 
# 29. score — баллы по госэкзамену по математике

# ### Рекомендации по выполнению проекта:

# 1. Проведите первичную обработку данных. Так как данных много, стоит написать функции, которые можно применять к столбцам определённого типа.
# 2. Посмотрите на распределение признака для числовых переменных, устраните выбросы.
# 3. Оцените количество уникальных значений для номинативных переменных.
# 4. По необходимости преобразуйте данные
# 5. Проведите корреляционный анализ количественных переменных
# 6. Отберите не коррелирующие переменные.
# 7. Проанализируйте номинативные переменные и устраните те, которые не влияют на предсказываемую величину (в нашем случае — на переменную score).
# 8. Не забудьте сформулировать выводы относительно качества данных и тех переменных, которые вы будете использовать в дальнейшем построении модели.

# In[80]:


import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import ttest_ind

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.datasets import fetch_20newsgroups, load_files

# Standard plotly imports
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode

# Using plotly + cufflinks in offline mode
import cufflinks
import plotly.figure_factory as ff
import plotly.express as px
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

# Настройка глобальной темы cufflinks
cufflinks.set_config_file(world_readable=True, theme='pearl', offline=True)
warnings.filterwarnings('ignore')


# ### Загружаем данные:

# In[81]:


data = pd.read_csv('StudMath.csv')
data.head(5)


# Посмотрим на размер массива, и предварительную информацию о нашем датасете.

# In[82]:


print(f'Размер датасета: {data.shape[0]} - строк, {data.shape[1]} - колонок')


# Посмотрим на название колонок:

# In[83]:


data.columns


# В датасете сразу бросается в глаза не соответствие количество входных признаков с описанием. Колонка (studytime, granular) не указана в описании и имеет отрицательные значения. Сравним значения колонок studytime и (studytime, granular):

# In[84]:


data[['studytime','studytime, granular']].head(5)


# In[85]:


fig = px.bar(data[['studytime','studytime, granular','score']], x=['studytime','studytime, granular'], y='score')
fig.show()


# Похоже что признак 'studytime, granular' является изменными значениями признака studytime. Визуально складывается впечатление что значения колонки 'studytime, granular' это умноженные на -3 значения studytime. Проверим это:

# In[86]:


(data['studytime'] - abs(data['studytime, granular']/3)).unique()


# Наше предположение оказалось абсолютно верным. В результате мы можем удалить столбец 'studytime, granular':

# In[87]:


data.drop(['studytime, granular'], inplace=True,axis=1)


# In[88]:


data.head(5)


# Посмотрим на количество отсутствующих значений и их процентное соотношение:

# In[89]:


def table(df):
    x = pd.DataFrame()
    x['Всего NaN'] = df.isna().sum()
    x['% NaN'] = round((x['Всего NaN'] / df.shape[0])*100, 2)
    return x.sort_values('% NaN', ascending=False)


table(data)


# Проще сказать где нет Nan )))

# Прежде чем приступить к непосредственному анализу данных их очистки и приведению к удобному для дальнейшей обработки виду, их следует разделить. Признаки в датасете можно разделить на две категории: количественные(числовые) и номинативные(категориальные). Для каждой категории у нас будут определенные методы анализа и восстановления данных.

# In[90]:


categorical_col = [
    col for col in data.columns if data[col].dtype.name == 'object']
numerical_col = [
    col for col in data.columns if data[col].dtype.name != 'object']

print(f'Количество категориальных признаков: {len(categorical_col)}')
print(categorical_col)
print(60*'==')
print(f'Количество числовых признаков: {len(numerical_col)}')
print(numerical_col)


# ## Рассмотрим количественные признаки:

# Для ускорения процессов анализа данных напишем несколько функций:
#     1. ColumnInfo() - эта функция будет выдавать информацию о количестве уникальных значений указанного признака и его распределении, также она показывает количество пропущенных значений и строит график распределения величины текущей (неизмененной) выборки признака;
#     2. ColumnApdateInfo() - строит график распределения величины текущей выборки признака после введеных изменнений;
#     3. FillNan() - заполняет отсутствующие значения колонки признака ее средним значением.

# In[91]:


def ColumnInfo(col):
    print(f'Список уникальных значений колонки: {col.unique()}')
    print(60*'==')
    print(f'Распределение данных: {col.describe()}')
    print(60*'==')
    print(f'Количество пропущенных ячеек: {data.shape[0]-col.count()}')
    df = col.value_counts()
    dt = df.reset_index()
    dt.rename(columns={dt.columns[0]: dt.columns[1],
                       dt.columns[1]: 'quantity'}, inplace=True)
    fig = px.bar(dt, x=dt.columns[0], y='quantity',
                 color='quantity', height=450, width=800)
    fig.show()


def ColumnApdateInfo(x):
    print(f'Количество пропущенных ячеек: {data.shape[0]-x.count()}')
    df = x.value_counts()
    dt = df.reset_index()
    dt.rename(columns={dt.columns[0]: dt.columns[1],
                       dt.columns[1]: 'quantity'}, inplace=True)
    fig = px.bar(dt, x=dt.columns[0], y='quantity',
                 color='quantity', height=450, width=800)
    fig.show()


def FillNan(coll):
    column_means = round(coll.mean(), 0)
    coll = coll.fillna(column_means, inplace=True)
    return coll


# In[92]:


data[numerical_col].describe()


# Из таблицы полученных значении видно, что только в столбце 'age' нет пропущенных значений. В столбцах 'Fedu', 'absences' и 'famrel' похоже есть выбросы, изучим каждую колонку более подробно:

# ### Колонка Age (возраст ученика):

# In[93]:


ColumnInfo(data.age)


# В колонке Age нет пропущенных значений и аномальных выбросов значений. Вопросы могли бы вызвать значения возрастной группы от 18 лет, но в самом задании сказано: отследить влияние условий жизни учащихся в возрасте от 15 до 22 лет. Все значения попадают в заданный интервал.

# ### Колонка Medu:
# Образование матери (0 - нет, 1 - 4 класса, 2 - 5-9 классы, 3 - среднее специальное или 11 классов, 4 - высшее)

# In[94]:


ColumnInfo(data.Medu)


# Заполнить пропущенные значения можно с помощью метода fillna. Заполним, например, медианными значениями и и округлим до целого числа.

# In[95]:


FillNan(data.Medu)
ColumnApdateInfo(data.Medu)


# В колонке Medu устранены пропущенные значения, аномальных выбросов значений нет. Исходя из полученного графика большинство матерей имеют среднее или среднее-специальное образование. 

# ### Колонка Fedu:
# Образование отца (0 - нет, 1 - 4 класса, 2 - 5-9 классы, 3 - среднее специальное или 11 классов, 4 - высшее)

# In[96]:


ColumnInfo(data.Fedu)


# Имеется выброс равный 40 скорее всего произошла опечатка и возможное значение это 4. Заменим это значение.

# In[97]:


data.Fedu = data.Fedu.apply(lambda x: 4 if x == 40 else x)


# Заполним пропущенные значения:

# In[98]:


FillNan(data.Fedu)
ColumnApdateInfo(data.Fedu)


# В колонке Fedu устранены пропущенные значения и аномальные выбросы. Исходя из полученного графика большинство отцов имеют среднее или среднее-специальное образование. 

# ### Колонка traveltime:
# Время в пути до школы (1 - <15 мин., 2 - 15-30 мин., 3 - 30-60 мин., 4 - >60 мин.)

# In[99]:


ColumnInfo(data.traveltime)


# Заполним пропущенные значения:

# In[100]:


FillNan(data.traveltime)
ColumnApdateInfo(data.traveltime)


# У большинства учащихся время до школы занимает не более 15 мин.

# ### Колонка studytime:
# Время на учёбу помимо школы в неделю (1 - <2 часов, 2 - 2-5 часов, 3 - 5-10 часов, 4 - >10 часов)

# In[101]:


ColumnInfo(data.studytime)


# Заполним пропущенные значения:

# In[102]:


FillNan(data.studytime)
ColumnApdateInfo(data.studytime)

Учащиеся тратят в среднем 2-5 часов на учебу во внеучебное время.
# ### Колонка failures:
# Количество внеучебных неудач (n, если 1<=n<3, иначе 0)

# In[103]:


ColumnInfo(data.failures)


# Заполним пропущенные значения:

# In[104]:


FillNan(data.failures)
ColumnApdateInfo(data.failures)


# ### Колонка famrel:
# Cемейные отношения (от 1 - очень плохо до 5 - очень хорошо)

# In[105]:


ColumnInfo(data.famrel)


# In[106]:


data.famrel = data.famrel.apply(lambda x: 1 if x == -1 else x)


# Заполним пропущенные значения:

# In[107]:


FillNan(data.famrel)
ColumnApdateInfo(data.famrel)


# ### Колонка freetime:
# Cвободное время после школы (от 1 - очень мало до 5 - очень мого)

# In[108]:


ColumnInfo(data.freetime)


# Заполним пропущенные значения:

# In[109]:


FillNan(data.freetime)
ColumnApdateInfo(data.freetime)


# ### Колонка goout:
# Проведение времени с друзьями (от 1 - очень мало до 5 - очень много)

# In[110]:


ColumnInfo(data.goout)


# Заполним пропущенные значения:

# In[111]:


FillNan(data.goout)
ColumnApdateInfo(data.goout)


# ### Колонка health:
# Текущее состояние здоровья (от 1 - очень плохо до 5 - очень хорошо)

# In[112]:


ColumnInfo(data.health)


# Заполним пропущенные значения:

# In[113]:


FillNan(data.health)
ColumnApdateInfo(data.health)


# ### Колонка absences:
# Количество пропущенных занятий

# In[114]:


ColumnInfo(data.absences)


# Самый простой способ отфильтровать выбросы — воспользоваться формулой интерквартильного расстояния (межквартильного размаха). Выбросом считаются такие значения, которые лежат вне рамок
# 
# percentile(25) -1.5*IQR : percentile(75)+1.5*IQR
# 
# IQR = percentile(75) - percentile(25).

# In[115]:


percintile_25 = 0
percintile_75 = 8
IQR = percintile_75 - percintile_25

Min = percintile_25 - 1.5*IQR
Max = percintile_75 + 1.5*IQR

if Min <= 0: Min = 0

print(f'Границы выбросов - {Min} : {Max}')


# In[116]:


data.absences = data.absences.apply(lambda x: None if x >= 20 else x)


# Заполним пропущенные значения:

# In[117]:


FillNan(data.absences)
ColumnApdateInfo(data.absences)


# ### Колонка score:
# Баллы по госэкзамену по математике

# In[118]:


ColumnInfo(data.score)


# Заполним пропущенные значения:

# In[119]:


FillNan(data.score)


# In[120]:


data[numerical_col].describe()


# ### Корреляционный анализ:

# Выясним, какие столбцы коррелируют с результатом на успеваемость по математике. Это поможет понять, какие параметры стоит оставить для модели, а какие — исключить. 
# 
# Корреляцию рассмотрим только для числовых столбцов:

# In[121]:


corrs = data.corr()

figure = ff.create_annotated_heatmap(
    z=corrs.values,
    x=list(corrs.columns),
    y=list(corrs.index),
    annotation_text=corrs.round(2).values,
    showscale=True)

figure.update_layout(
    autosize=False,
    width=1000,
    height=1000)

figure


# In[122]:


data.corr()[:-1]['score']


# ### Промежуточный вывод:
# Все значения кроме 'failures' имеют очень слабую корреляцию менее 0.2. Вполне логично что с уменьшением 'failures' значение 'score' будет увеличиваться. 
# Интересно обратить внимание на то что значения Medu и Fedu достаточно сильно коррелируют между собой, но при этом коэффициет корреляции Medu и score равен 0.21, а коэффициет корреляции Fedu и score равен 0.11. Можно предположить что семейные пары формируются в том числе исходя из уроня их образования, а возможно большинство пар просто знакомятся в момент учебы в колледже, университете итд. В любом слючае стоит заметить что уровень образования родителей имеет влияние на вероятность успешно сдать тест. Это еще раз доказывает отрицательная корреляция Medu и Fedu по отношению к 'failures', видимо родители помогают детям делать уроки или хотя бы дают дельные советы).

# Проверим, есть ли статистическая разница в распределении оценок по числовым признакам с помощью теста Стьюдента. 
# Проверим нулевую гипотезу о том, что распределения оценок результирующего теста по различным параметрам неразличимы:

# Учтем полученные результаты и составим финальный датасет для числовых признаков:

# In[123]:


numerical_for_new_date = data[['age', 'Medu',
                               'studytime', 'failures', 'goout', 'absences']]


# ## Рассмотрим номинативные колонки:

# In[124]:


data[categorical_col].describe()


# И в этом наборе данных проще перечислить признаки где отсутствуют пропущенные значения: school, sex. Заполним отсутствующие ячейки наиболее часто встречающимися значениями в признаках:

# In[125]:


data_describe = data.describe(include=[object])

for col in categorical_col:
    data[col] = data[col].fillna(data_describe[col]['top'])


# In[126]:


data.describe(include=[object])


# Очевидно, что для номинативных переменных использовать корреляционный анализ не получится. Однако можно посмотреть, различаются ли распределения score в зависимости от значения этих переменных. Это можно сделать с помощью box-plot (график показывает плотность распределения переменных).

# In[127]:


def BoxPlot(col):
    if data[col].dtype.name == 'object':
        fig = px.box(data_frame=data, x=col, y='score', color=col)
        fig.show()
    else:
        pass
    return


for col in data:
    BoxPlot(col)


# По графикам похоже, что все параметры, кроме activites, paid, reason могут влиять на результирующие оценки теста. Однако графики являются лишь вспомогательным инструментом, настоящую значимость различий может помочь распознать статистика. 
# Проверим, есть ли статистическая разница в распределении оценок по номинативным признакам, с помощью теста Стьюдента. 
# Проверим нулевую гипотезу о том, что распределения оценок результирующего теста по различным параметрам неразличимы:

# In[128]:


def get_stat_dif(column):
    cols = data.loc[:, column].value_counts().index
    combinations_all = list(combinations(cols, 2))
    for comb in combinations_all:
        if ttest_ind(data.loc[data.loc[:, column] == comb[0], 'score'],
                     data.loc[data.loc[:, column] == comb[1], 'score']).pvalue \
                <= 0.05/len(combinations_all):  # Учли поправку Бонферони
            print('Найдены статистически значимые различия для колонки', column)
            break


# In[129]:


for col in data[categorical_col]:
    get_stat_dif(col)


# Как мы видим, серьёзно отличаются четыре параметра: address,  Mjob, higher  и  romantic. Оставим эти переменные в датасете для дальнейшего построения модели.

# In[130]:


nominative_for_new_date = data[['address', 'Mjob', 'higher', 'romantic']]
nominative_for_new_date.describe()


# ### Сформируем итоговый датасет:

# In[131]:


Total_data = pd.concat(
    (numerical_for_new_date, nominative_for_new_date), axis=1)
Total_data


# ## Вывод:

# Итак, в результате EDA для анализа условий жизни учащихся в возрасте от 15 до 22 лет на их успеваемость по математике были получены следующие выводы:
# 
# В данных было достаточно пустых значений, выбросы найдены только в столбцах (Fedu - образование отца) и (famrel - отношения в семье), также выбросы устранены в столбце пропущенных занятий.
# Пропущенные значения заполнялись средними значениями вычисленные для каждого числового столбца и наиболее часто встречающимися значениями для номинативных значениях.
# 
# Корреляционный анализ предоставил следующие закономерности:
# 1. Все значения кроме 'failures' имеют очень слабую корреляцию менее 0.2;
# 2. Значения Medu и Fedu достаточно сильно коррелируют между собой;
# 3. Отрицательная корреляция Medu и Fedu по отношению к 'failures';
# 4. "Нелогичная" (на мой взгляд) корреляция 'absences'- прогулов к результату теста score. Всего 0.07 и она положительна!
# 5. Достаточно высокая (в сравнении с другими признаками) корреляция 'age' и 'score'. Чем младше ученики тем лучше их результаты.
# 
# В итоге были оставлены признаки со значениями корреляции выше 0.1.
# 
# Для номинативных данных была расчитана статистическая разница в распределении оценок по числовым признакам с помощью теста Стьюдента. Проверили нулевую гипотезу о том, что распределения оценок результирующего теста по различным параметрам неразличимы,  в результате получили 4 столбца, где статистически значимые различия были найдены. Столбцы ['address', 'Mjob', 'higher', 'romantic'].
# 
# Эти операции позволили нам очистить и привести данные к дальнейшему анализу.

# -------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------

# # Машинное обучение:

# ## Векторизация:

# Библиотека scikit-learn не умеет напрямую обрабатывать категориальные признаки. Поэтому прежде чем подавать данные на вход алгоритмов машинного обучения преобразуем категориальные признаки в количественные.
# 
# Категориальные признаки, принимающие два значения (т.е. бинарные признаки) и принимающие большее количество значений будем обрабатывать по-разному.

# In[132]:


data_describe = nominative_for_new_date.describe(include=[object])

binary_columns = [
    col for col in nominative_for_new_date if data_describe[col]['unique'] == 2]
nonbinary_columns = [
    col for col in nominative_for_new_date if data_describe[col]['unique'] > 2]

print(f'Колонки с бинарными признаками: {binary_columns}')
print(60*'==')
print(f'Колонки с небинарными признаками:{nonbinary_columns}')


# #### Бинарные признаки:

# In[133]:


def Binary_convert(col):
    first = Total_data[col].unique()[0]
    second = Total_data[col].unique()[1]

    nominative_for_new_date.at[Total_data[col] == first, col] = 1
    nominative_for_new_date.at[Total_data[col] == second, col] = 0
    return


# In[134]:


for col in nominative_for_new_date[binary_columns]:
    Binary_convert(col)


# In[135]:


nominative_for_new_date.head(5)


# #### Небинарные признаки:

# К небинарными признакам применим метод векторизации. Такую векторизацию осуществляет в pandas метод get_dummies:

# In[136]:


nonbinary_data = pd.get_dummies(nominative_for_new_date[nonbinary_columns])


# In[137]:


nonbinary_data.head(5)


# In[138]:


nominative_for_new_date.drop(['Mjob'], inplace=True, axis=1)
nominative_for_new_date.head(5)


# ###  Соединим все столбцы в одну таблицу:

# In[139]:


ML_data = pd.concat(
    (numerical_for_new_date, nominative_for_new_date, nonbinary_data), axis=1)


# In[140]:


ML_data.head(5)


# ### Разбиваем модель:

# In[141]:


X = ML_data.iloc[:, :].values
y = data.iloc[:, -1].values


# Сформируем массив для выделенного массива 'score'. В нашем прогнозе зададимся простым условием: сможет ли учащийся сдать тест. Соответственно 1 - сдаст и 0 - не сдаст. Изучив методику оценивания в США и Европе приходим к выводу что минимальное количество для сдачи теста является 49, это и будет нашим граничным значением.

# In[142]:


y = []
for i in data['score']:
    if i < 49:
        y.append(0)
    else:
        y.append(1)


# In[143]:


from sklearn.model_selection import train_test_split

s = train_test_split(X, y, test_size=0.1, stratify=y, random_state=1)
X_train, X_test, y_train, y_test = s


# ### Масштабируем признаки:

# In[144]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler().fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# # Алгоритмы машинного обучения:

# In[145]:


Result = []


# In[146]:


from sklearn.metrics import accuracy_score


def log_errors(val_train, val_test):
    y_train_pred = val_train
    y_test_pred = val_test

    num1 = (y_train != y_train_pred).sum()
    num2 = (y_test != y_test_pred).sum()

    acc1 = round(accuracy_score(y_train, y_train_pred)*100, 2)
    acc2 = round(accuracy_score(y_test, y_test_pred)*100, 2)
    d = {'Train': acc1, 'Test': acc2}
    Result.append(d)
    return print(f'Кол-во ошибок обучающей выборки = {num1} \n',
                 f'Кол-во ошибок тестовой выборки = {num2} \n',
                 f'Оценка точности обучающей выборки = {acc1} % \n',
                 f'Оценка точности тестовой выборки = {acc2} %', sep='\n')


# # Логистическая регрессия:

# In[147]:


from sklearn.linear_model import LogisticRegression

LgReg = LogisticRegression(random_state  = 2, C = 10.0).fit(X_train_std, y_train)
log_errors(LgReg.predict(X_train_std), LgReg.predict(X_test_std))


# # Метод опорных векторов:  

# ### Выберем ядро RBF:

# In[148]:


from sklearn.svm import SVC
from sklearn.model_selection import learning_curve, GridSearchCV

C_array = np.logspace(-3, 3, num=7)
gamma_array = np.logspace(-5, 2, num=8)
svc = SVC(kernel='rbf')

grid = GridSearchCV(svc, param_grid={'C': C_array, 'gamma': gamma_array}).fit(
    X_train, y_train)

print('CV error    = ', 1 - grid.best_score_)
print('best C      = ', grid.best_estimator_.C)
print('best gamma  = ', grid.best_estimator_.gamma)


# In[149]:


svc1 = SVC(kernel='rbf', C=grid.best_estimator_.C,
           gamma=grid.best_estimator_.gamma).fit(X_train, y_train)
log_errors(svc1.predict(X_train_std), svc1.predict(X_test_std))


# ### Линейное ядро:

# In[150]:


C_array = np.logspace(-3, 3, num=7)
svc = SVC(kernel='linear')
grid = GridSearchCV(svc, param_grid={'C': C_array}).fit(X_train, y_train)

print('CV error    = ', 1 - grid.best_score_)
print('best C      = ', grid.best_estimator_.C)


# In[151]:


svc2 = SVC(kernel='linear', C=grid.best_estimator_.C).fit(X_train, y_train)
log_errors(svc2.predict(X_train_std), svc2.predict(X_test_std))


# # Метод ближайших соседей:

# In[152]:


from sklearn.neighbors import KNeighborsClassifier

nei1 = KNeighborsClassifier(n_neighbors=3).fit(X_train_std, y_train)
log_errors(nei1.predict(X_train_std), nei1.predict(X_test_std))


# In[153]:


from sklearn.neighbors import KNeighborsClassifier

nei2 = KNeighborsClassifier(n_neighbors=7).fit(X_train_std, y_train)
log_errors(nei2.predict(X_train_std), nei2.predict(X_test_std))


# # AdaBoost – адаптивный бустинг:

# In[154]:


from sklearn import ensemble

ada = ensemble.AdaBoostClassifier(n_estimators=50, learning_rate=1,
                                  algorithm='SAMME.R').fit(X_train, y_train)
log_errors(ada.predict(X_train_std), ada.predict(X_test_std))


# # Tree - дерево решений:

# In[155]:


from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=10).fit(X_train, y_train)
log_errors(tree.predict(X_train_std), tree.predict(X_test_std))


# # Random Forest – случайный лес:

# In[156]:


from sklearn import ensemble

rf = ensemble.RandomForestClassifier(criterion='gini', max_depth=3,
                                     min_samples_split=2, min_samples_leaf=2).fit(X_train, y_train)
log_errors(rf.predict(X_train_std), rf.predict(X_test_std))


# # GBT – градиентный бустинг деревьев решений:

# In[157]:


from sklearn import ensemble

gbt = ensemble.GradientBoostingClassifier(
    n_estimators=50, random_state=3).fit(X_train, y_train)
log_errors(gbt.predict(X_train_std), gbt.predict(X_test_std))


# In[159]:


name = {'Algoritm': ['Logistic', 'SVC_RBF', 'SVC_linear',
                     'KNN_n=3', 'KNN_n=7', 'Adaboost', 'Tree', 'Rand_Forest', 'GBT']}

Alg = pd.DataFrame(data=name)
ML = pd.DataFrame(data=Result)

Res = pd.concat((Alg, ML), axis=1)
Res


# Матрица ошибок у KNN n=3 показывает, что эта модель - лидер по поиску истинноположительных ответов. Это говорит о том, что она с большей вероятностью по сравнению с остальными моделями сможет определить учеников, которые сдадут тест по математике.
