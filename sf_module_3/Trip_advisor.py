#!/usr/bin/env python
# coding: utf-8

# In[561]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[562]:


RANDOM_SEED = 42
get_ipython().system('pip freeze > requirements.txt')


# In[563]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import re
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 

get_ipython().run_line_magic('matplotlib', 'inline')


# # **Import DATA**

# In[564]:


DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'
df_train = pd.read_csv(DATA_DIR+'/main_task.csv')
df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')
sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')


# In[565]:


df_train['train'] = 1 # помечаем где у нас трейн
df_test['train'] = 0 # помечаем где у нас тест
df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями

data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем


# Подробнее по признакам:
# 
# 1. City: Город
# 2. Cuisine Style: Кухня
# 3. Ranking: Ранг ресторана относительно других ресторанов в этом городе
# 4. Price Range: Цены в ресторане в 3 категориях
# 5. Number of Reviews: Количество отзывов
# 6. Reviews: 2 последних отзыва и даты этих отзывов
# 7. URL_TA: страница ресторана на 'www.tripadvisor.com'
# 8. ID_TA: ID ресторана в TripAdvisor
# 9. Rating: Рейтинг ресторана

# # Обработка

# ## **1. Первичная обработка данных**
# У наличия пропусков могут быть разные причины, но пропуски нужно либо заполнить, либо исключить из набора полностью. Но с пропусками нужно быть внимательным, даже отсутствие информации может быть важным признаком!
# По этому перед обработкой NAN лучше вынести информацию о наличии пропуска как отдельный признак

# In[566]:


# Рассмотрим названия колонок во избежании дальнейших трудностей при обработке данных:
data.columns


# In[567]:


# Названия колонок корректны, но для большего удобства сведем все названия к нижнему регистру:
dt = data.copy()
dt.rename(columns={'Restaurant_id':'res_id','City':'city','Cuisine Style':'cuis_style',
                  'Ranking':'rank','Price Range':'price','Number of Reviews':'num_rev',
                  'Reviews':'review','Rating':'rating',
                  'URL_TA':'url','ID_TA':'idta'}, inplace=True)


# In[568]:


# Посмотрим на размер массива, и предварительную информацию о нашем датасете.
print(f'Размер датасета: {dt.shape[0]} - строк, {dt.shape[1]} - колонок')


# In[569]:


dt.info()


# In[570]:


# убираем не нужные для модели признаки
dt.drop(['res_id','url','idta',], axis = 1, inplace=True)


# In[571]:


# Посмотрим на количество отсутствующих значений и их процентное соотношение:

def table(df):
    x = pd.DataFrame()
    x['Всего NaN'] = df.isna().sum()
    x['% NaN'] = round((x['Всего NaN'] / df.shape[0])*100, 2)
    return x.sort_values('% NaN', ascending=False)

table(dt)


# Итак в 4 столбцах нашего датасета есть пропущенные значения. 

# Прежде чем приступить к непосредственному анализу данных их очистки и приведению к удобному для дальнейшей обработки виду, их следует разделить. Признаки в датасете можно разделить на две категории: количественные(числовые) и номинативные(категориальные). Для каждой категории у нас будут определенные методы анализа и восстановления данных.

# In[572]:


categorical_col = [
    col for col in dt.columns if dt[col].dtype.name == 'object']
numerical_col = [
    col for col in dt.columns if dt[col].dtype.name != 'object']

print(f'Количество категориальных признаков: {len(categorical_col)}')
print(categorical_col)
print(60*'==')
print(f'Количество числовых признаков: {len(numerical_col)}')
print(numerical_col)


# In[573]:


# Рассмотрим количественные признаки:
dt[numerical_col].describe()


# In[574]:


plt.rcParams['figure.figsize'] = (10,7)
dt['rank'].hist(bins=100)


# У нас много ресторанов, которые не дотягивают и до 2500 места в своем городе, а что там по городам?

# In[575]:


dt['city'].value_counts(ascending=True).plot(kind='barh')


# Посмотрим, как изменится распределение в большом городе:

# In[576]:


dt['rank'][dt['city'] =='London'].hist(bins=100)


# In[577]:


# посмотрим на топ 10 городов
for x in (df_train['City'].value_counts())[0:10].index:
    df_train['Ranking'][df_train['City'] == x].hist(bins=100)
plt.show()


# Получается, что Ranking имеет нормальное распределение, просто в больших городах больше ресторанов, из-за мы этого имеем смещение.

# ## 2. Обработка категориальных признаков:

# In[578]:


def tokenizer(col):
    # Получим список всех значений в данном признаке:
    vocab = []
    for row in dt[col]:
        vocab.append(str(row))
    # Создаем список токенов:        
    token = sorted(set(vocab))
    # Задаем словарь с числовым значением для каждого токена:
    dictionary = {elem:ind for ind, elem in enumerate(token)}
    # Применяем словарь к исходной колонке:
    return dictionary


# ### Обработка NAN
# ![](http://)Создадим столбцы с информацией о том, где были пропуски.

# In[579]:


dt['cuis_nan'] = pd.isna(dt['cuis_style']).astype('uint8')
dt['price_nan'] = pd.isna(dt['price']).astype('uint8')


# ### Price:

# Определим все возможные значения в столбце и присвоим им числовые определители.

# In[580]:


dt['price'].value_counts()


# In[581]:


# Создадим числовые признаки  для колонки Price:
token = tokenizer('price')
# Применим полученный словарь к столбцу
dt = dt.replace({'price': token})


# In[582]:


# Заполним Nan
dt['price'].fillna(3, inplace=True)


# ### Number of reviews:

# Рассмотрим количество отзывов по городам:

# In[583]:


num_rev_per_city = dt.groupby(['city'])['num_rev'].sum().sort_values(ascending=True)


# In[584]:


num_rev_per_city.plot(kind='barh')


# Добавим признак ранга относительно количества отзывов для каждого города:

# In[585]:


dt['rev_per_city'] = dt['city'].apply(lambda row: num_rev_per_city[row])
dt['rel_rank'] = dt['rank'] / dt['rev_per_city']
dt.drop(['rev_per_city'], axis = 1, inplace=True)


# ### City:

# In[586]:


# Cоздадим словарь столиц европейских стран нашего датасета
capitals = {'London':1,'Paris':1,'Madrid':1,'Barcelona':0,'Berlin':1,'Milan':0,'Rome':1,'Prague':1,        
'Lisbon':1,'Vienna':1, 'Amsterdam':1,'Brussels':1,'Hamburg':0,'Munich':0,'Lyon':0,'Stockholm':1,     
'Budapest': 1,'Warsaw':1,'Dublin':1,'Copenhagen':1,'Athens':1,'Edinburgh':0,'Zurich':0,'Oporto':0,         
'Geneva':0,'Krakow':0,'Oslo':1,'Helsinki':1,'Bratislava':1,'Luxembourg':1,'Ljubljana':1}
# Создадим новый признак является ли город столицей страны:
dt['capital'] = dt['city']
# Применим словарь к новому признаку
dt = dt.replace({'capital': capitals})


# In[587]:


token = tokenizer('city')
# Применим полученный словарь к столбцу
dt = dt.replace({'city': token})


# In[588]:


# Создадим признак количества ресторанов для каждого города
rest_in_city = dt.groupby('city')['rank'].count().to_dict()
dt['rest_count'] = dt['city'].map(rest_in_city)


# ### Number of review

# In[589]:


nan = dt['num_rev'].isna().sum()
print(f'Количество пропущенных значений: {nan}')


# In[590]:


plt.rcParams['figure.figsize'] = (10,7)
dt['num_rev'].hist(bins=30)


# In[591]:


#Заполним пропущенные значения нулями
dt['num_rev'] = dt['num_rev'].fillna(0)


# ### Reviews:

# In[592]:


# заполним пропуски неопределенным значением [[], []] и создадим новый признак
# где есть пропуски в review 
dt['review'] = dt['review'].fillna('[[], []]')
dt['review_nan'] = (dt['review']=='[[], []]').astype('float64')


# In[593]:


# вытащим дату из ревью и создадим новые критерии
dt['date'] = dt['review'].str.findall('\d+/\d+/\d+')
dt['len_date'] = dt['date'].apply(lambda x: len(x))


# In[594]:


# создадим признак количества дней с последнего отзыва до текущей даты
day_to_now = []

for i in dt['date']:
    if len(i) == 0:
        day_to_now.append(0)
    else:
        day_to_now.append((datetime.datetime.now() - pd.to_datetime(i).max()).days)

dt['day_to_now'] = day_to_now


# In[595]:


# создадим признак количества дней между отзывами
# если отзыв один то значение 0 то же и для больших значений
review_between = []

for i in dt['date']:
    if len(i) == 2:
        review_between.append((pd.to_datetime(i).max() - pd.to_datetime(i).min()).days)
    else:
        review_between.append(0)
        
dt['review_between'] = review_between


# In[596]:


# обработаем сами отзывы, очистим мусор и создадим три списка:
# review - очищенная строка отзыва (понадобится в будущем)
# num_review - список отзывов для каждого ресторана
# review_list - список списков слов в отзыве для поиска положительных и 
# негативных отзывов

review = []
num_review = []
review_list = []

for row in dt['review']:
    row = str(row).replace("'",'"').replace('[["','').replace('", "',' $ ')
    row = str(row).replace('"], ["','|')
    row = row.lower()
    ind_num = row.find("|")
    row = row[:ind_num]
    row = str(row).replace('[[], []','None')
    rew = row.split('$')
    num_review.append(rew)
    row = str(row).replace('$ ','').replace('!','').replace(',','').replace('...','').replace('.','')
    row = str(row).replace(' - ',' ').replace(' :','').replace('"','').replace(' & ',' ').replace(' :)','')
    review.append(row)
    row = row.split(' ')
    review_list.append(row)


# In[597]:


# создаим список количества отзывов для ресторана
numbers = []

for i in num_review:
    if 'None' in i:
        numbers.append(int(0))
    else:
        numbers.append(int(len(i)))


# In[598]:


# задаем новые признаки
dt['number_review'] = numbers
dt['review'] = review


# In[599]:


# зададим пару списков с набором положительных и отрицательных слов, которые
# могут содержаться в отзывах

good_rew = ['good', 'fine', 'better', 'exquisite', 'best', 'supper', 'great', 
            'welcome', 'nice', 'tasty', 'high quality', 'low price', 'magnificent',
           'delicious', 'recommended', 'excellent', 'amazing', 'perfect', 'treasure', 
            'yummy', 'wonderful', 'breathtaking', 'ok', 'okay', 'not bad']

bad_rew = ['nightmare', 'worst', 'bad','sad', 'disgusting', 'rip', 'bad', 'disappointing', 'sadly', 'unclear',
           'dull', 'terrible', 'forget it', 'worth', 'awful', 'avoid', 'not good', 'slow', 'serious attention', 
           'worse', 'not fantastic','horrible', 'tragic', 'avoid', 'unfortunate']


# In[600]:


# 0 - нет или неопределенный комментарий по поводу ресторана
# 1 - положительный комментарий по поводу ресторана
# 2 - отрицательный комментарий по поводу ресторана

# Туповатый способ определить тип комментария: считаем вхождение положительный и
# негативных слов в строку, их разность определяет тип комментария
rev_dict = []

for i in review_list:
    good_review=list(set(good_rew) & set(i))
    bad_review=list(set(bad_rew) & set(i))
    result = len(good_review) - len(bad_review)
    if result == 0:
        rev_dict.append(0)
    elif result > 0:
        rev_dict.append(1)
    else:
        rev_dict.append(2)
        
dt['review'] = rev_dict


# ### Raiting:

# In[601]:


# ничего интересного 0 соответствует тестовым значениям


# In[602]:


plt.rcParams['figure.figsize'] = (10,7)
dt['rating'].hist(bins=30)


# In[603]:


def round_of_rating(number):
    return np.round(number * 2) / 2


# # Cuisine

# * Колонка Cuisine Style:
# * Количество пропущенных значений признака: 11590
# * Процент пропущенных значений признака: 23.18 %

# Пропущенных значений слишком много, поэтому заполним отсутсующие значения признака новым.

# In[604]:


# заполним пропущенные значения параметром other и отберем топ 30 кухонь для обработки
dt['cuis_style'].fillna('other', inplace = True)
df = dt['cuis_style'].value_counts().head(30)
df


# In[605]:


# создадим два новых признака 
# cuis_num - количество кухонь для ресторана
# cuis_style - тип кухни если не входит в топ 30 получает маркер other

cuis_style = []
cuis_num = []

for row in dt['cuis_style']:
    if row in df:
        row = str(row).replace("['","").replace("']","").replace("', '","|")
        cuis_style.append(row)
    else:
        cuis_style.append('other')
    row = row.split('|')
    cuis_num.append(len(row))
    
dt['cuis_style'] = cuis_style
dt['cuis_num'] = cuis_num


# In[606]:


# Создадим числовые признаки  для колонки Price:
token = tokenizer('cuis_style')
# Применим полученный словарь к столбцу
dt = dt.replace({'cuis_style': token})


# ### Корреляционный анализ

# In[607]:


plt.rcParams['figure.figsize'] = (15,15)
sns.heatmap(dt.corr(), square=True,
            annot=True, fmt=".1f", linewidths=0.1, cmap="RdBu");
plt.tight_layout()


# Столбцы 'cuis_nan','price_nan','review_nan' сильно коррелируют с родительскими столбцами поэтому смысла в них нет удаляем. Столбец date удаляем так как он не несет более никакой полезной информации.

# In[608]:


dt.drop(['date','cuis_nan','price_nan','review_nan'], axis = 1, inplace=True)


# Оценим распределение величин в нашем датасете

# In[609]:


dt[dt.columns].hist(figsize=(20, 20), bins=10);
plt.tight_layout()


# ### Нормализуем признаки:

# In[610]:


# в датасете величины имеют сильный разброс от единиц до нескольких тысяч
# для корректной работы алгоритма следует привести их к соответствующим значениям

def norn(col):
    xnorm = []
    xmax = dt[col].describe()[-1]
    xmin = dt[col].describe()[3]
    for x in dt[col]:
        xnorm.append((x - xmin)/(xmax - xmin))
    dt[col] = xnorm
    return dt[col]


# In[611]:


for col in dt.columns:
    if col != 'rating':
        norn(col)
    else:
        pass


# In[612]:


dt.sample(10)


# ### Разбиваем датасет на тренировочный и тестовый

# In[613]:


train_data = dt.query('train == 1').drop(['train'], axis=1)
test_data = dt.query('train == 0').drop(['train'], axis=1)

y = train_data.rating.values         
X = train_data.drop(['rating'], axis=1)


# In[614]:


# используем train_test_split для разбивки тестовых данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)


# In[615]:


# проверяем размеры массивов
test_data.shape, train_data.shape, X.shape, X_train.shape, X_test.shape


# ### Model

# In[616]:


# Импортируем необходимые библиотеки:
from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели
from sklearn import metrics # инструменты для оценки точности модели


# In[617]:


# Создаём модель (НАСТРОЙКИ НЕ ТРОГАЕМ)
model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)


# In[618]:


# Обучаем модель на тестовом наборе данных
model.fit(X_train, y_train)

# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.
# Предсказанные значения записываем в переменную y_pred
y_pred = model.predict(X_test)


# In[619]:


y_pred_old = y_pred.copy()
y_pred = round_of_rating(y_pred) 


# In[620]:


# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются
# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.
print('MAE:', metrics.mean_absolute_error(y_test, y_pred),  metrics.mean_absolute_error(y_test, y_pred_old) )


# In[621]:


#Вычисляем коэффициент детерминации:
R_2 = metrics.r2_score(y_test, y_pred)
print(R_2)


# In[622]:


# в RandomForestRegressor есть возможность вывести самые важные признаки для модели
plt.rcParams['figure.figsize'] = (10,10)
feat_importances = pd.Series(regr.feature_importances_, index=X.columns)
feat_importances.nlargest(30).plot(kind='barh');


# # Submission

# In[623]:


test_data.sample(10)


# In[624]:


test_data = test_data.drop(['rating'], axis=1)


# In[625]:


sample_submission.head(10)


# In[626]:


predict_submission = model.predict(test_data)


# In[627]:


predict_submission = np.round(predict_submission, 1)


# In[628]:


sample_submission['Rating'] = predict_submission
sample_submission.to_csv('submission.csv', index=False)
sample_submission.head(10)

