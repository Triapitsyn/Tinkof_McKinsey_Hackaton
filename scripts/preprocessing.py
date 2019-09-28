# В этом файле будут находиться функции для препроцессинга датасетов. Сейчас
# готов только датасет customer.

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np


# Функция для LabelEncoding столбцов со строковыми значениями
def label_encode_customer(cust):
    columns_to_encode = [f'product_{i}' for i in range(7)] + ['marital_status_cd', 'job_title']
    for col in columns_to_encode:
        cust[col].fillna('nan', inplace=True)
        cust[col] = LabelEncoder().fit_transform(cust[col])
    return cust


# Функция, в которой мы заполняем NaNы в некоторых столбцах медианой.
def fill_nans_customer(cust):
    columns_to_fill = ['age', 'first_session_year', 'first_session_month',
                       'first_session_day', 'first_session_hour']
    for col in columns_to_fill:
        cust[col].fillna(cust[col].median(), inplace=True)
    cust['job_position_cd'].fillna(-1, inplace=True)
    cust['children_cnt'].fillna(-1, inplace=True)
    return cust


# Функция preprocess_customer выполняет препроцессинг датасета customer. Сейчас можно выбрать
# один или несколько способов закодировать категориальные переменные - One Hot Encoding,
# Frequency encoding и mean encoding. Соответственно, в encodings надо передавать занчения 'mean',
# 'one-hot' или 'frequency'. Label Encoding будет по умолчанию.
def preprocess_customer(customer, react=None, encodings=[], drop_original=False, most_common_jobs=15):
    cust = customer.copy()

    # Обработка пола, Male - 1, Female - 0. NaNы заполняем как Male, потому что мужчин больше.
    cust['gender_cd'].fillna('M', inplace=True)
    cust['gender_cd'] = (cust['gender_cd'] == 'M').astype(int)

    # job_title переводим в нижний регистр
    cust['job_title'] = cust['job_title'].str.lower()

    # Обработка даты первой сессии
    cust['first_session_dttm_nan'] = (cust['first_session_dttm'].isna()).astype(int)
    cust['first_session_dttm'] = cust['first_session_dttm'].apply(pd.to_datetime)
    cust['first_session_year'] = pd.DatetimeIndex(cust['first_session_dttm']).year
    cust['first_session_month'] = pd.DatetimeIndex(cust['first_session_dttm']).month
    cust['first_session_day'] = pd.DatetimeIndex(cust['first_session_dttm']).day
    cust['first_session_hour'] = pd.DatetimeIndex(cust['first_session_dttm']).hour
    cust.drop('first_session_dttm', axis=1, inplace=True)

    # Обработка возраста
    cust['age_less_than_17'] = (cust['age'] < 17).astype(int)
    cust['age_17_25'] = ((cust['age'] >= 17) & (cust['age'] <= 25)).astype(int)
    cust['age_26_40'] = ((cust['age'] >= 26) & (cust['age'] <= 40)).astype(int)
    cust['age_more_than_40'] = (cust['age'] > 40).astype(int)
    cust['age_nan'] = (cust['age'].isna()).astype(int)

    # Кодируем и заполняем пропуски
    cust = label_encode_customer(cust)
    cust = fill_nans_customer(cust)

    for encoding in encodings:
        if encoding == 'one-hot':
            common_jobs = cust['job_title'].value_counts().index[:most_common_jobs]
            cust['job_title'] = cust['job_title'].apply(lambda x:
                                                        x if x in common_jobs else -1)

            columns_to_encode = [f'product_{i}' for i in range(7)] + ['marital_status_cd', 'job_title']
            for col in columns_to_encode:
                one_hot = pd.get_dummies(cust[col])
                one_hot.columns = [f'{col}_one_hot_{str(val)}' for val in one_hot.columns]
                cust = pd.concat([cust, one_hot], axis=1)

        if encoding == 'frequency':
            columns_to_encode = [f'product_{i}' for i in range(7)] + ['marital_status_cd', 'job_title',
                                                                      'first_session_year', 'first_session_month',
                                                                      'first_session_day', 'first_session_hour', 'age',
                                                                      'children_cnt', 'job_position_cd']
            for col in columns_to_encode:
                vc = cust[col].value_counts()
                cust = cust.join(vc, on=col, rsuffix='_frequency_encoded')
        if encoding == 'mean':
            columns_to_encode = [f'product_{i}' for i in range(7)] + ['marital_status_cd', 'job_title',
                                                                      'first_session_year', 'first_session_month',
                                                                      'first_session_day', 'first_session_hour', 'age',
                                                                      'children_cnt', 'job_position_cd']

            event_encoded = pd.get_dummies(react['event'])
            for col in columns_to_encode:
                joint = pd.concat([react[['customer_id']], event_encoded], axis=1)
                joint = joint.join(cust[[col, 'customer_id']].set_index('customer_id'),
                                   on='customer_id')
                joint.drop('customer_id', axis=1, inplace=True)
                joint.columns = [name + '_to_mean_' + col for name in joint.columns]
                cust = cust.join(joint.groupby(col + '_to_mean_' + col).mean())
            cust.fillna(0., inplace=True)
    if drop_original:
        cols_to_drop = [f'product_{i}' for i in range(7)] + ['marital_status_cd', 'job_title']
        cust.drop(cols_to_drop, axis=1, inplace=True)

    return cust


# Выполняет препроцессинг таблицы реакций на истории. Позволяет использовать frequency, mean
# и std encodings. Также позволяет энкодить ивент как one-hot, label или никак (None).
def preprocess_reactions(train_react, test_react, encodings=[], drop_original=False,
                         encode_event='label', verbose=True):
    train = train_react.copy()
    test = test_react.copy()

    test.drop('answer_id', axis=1, inplace=True)

    train['event_dttm'] = train['event_dttm'].apply(pd.to_datetime)
    train['event_year'] = pd.DatetimeIndex(train['event_dttm']).year
    train['event_month'] = pd.DatetimeIndex(train['event_dttm']).month
    train['event_day'] = pd.DatetimeIndex(train['event_dttm']).day
    train['event_hour'] = pd.DatetimeIndex(train['event_dttm']).hour
    train.drop('event_dttm', axis=1, inplace=True)

    test['event_dttm'] = test['event_dttm'].apply(pd.to_datetime)
    test['event_year'] = pd.DatetimeIndex(test['event_dttm']).year
    test['event_month'] = pd.DatetimeIndex(test['event_dttm']).month
    test['event_day'] = pd.DatetimeIndex(test['event_dttm']).day
    test['event_hour'] = pd.DatetimeIndex(test['event_dttm']).hour
    test.drop('event_dttm', axis=1, inplace=True)

    event_encoded = pd.get_dummies(train['event'])
    train = pd.concat([train, event_encoded], axis=1)
    test['dislike'], test['like'], test['view'], test['skip'] = 0, 0, 0, 0  # Иначе путаница
    # с именами столбцов

    cols_to_encode = ['event_year', 'event_month', 'event_day', 'event_hour', 'customer_id', 'story_id']
    for encoding in encodings:
        if encoding == 'frequency':
            for col in cols_to_encode:
                vc = train[col].value_counts()
                train = train.join(vc, on=col, rsuffix='_frequency_encoded')
                test = test.join(vc, on=col, rsuffix='_frequency_encoded')
        if encoding == 'mean':
            for col in cols_to_encode:
                joint = pd.concat([train[[col]], event_encoded], axis=1)
                train = train.join(joint.groupby(col).mean(), on=col, rsuffix='_mean_encoded_' + col)
                test = test.join(joint.groupby(col).mean(), on=col, rsuffix='_mean_encoded_' + col)
        if encoding == 'std':
            for col in cols_to_encode:
                joint = pd.concat([train[[col]], event_encoded], axis=1)
                train = train.join(joint.groupby(col).std(), on=col, rsuffix='_std_encoded_' + col)
                test = test.join(joint.groupby(col).std(), on=col, rsuffix='_std_encoded_' + col)
    if drop_original:
        train.drop(cols_to_encode, axis=1, inplace=True)
        test.drop(cols_to_encode, axis=1, inplace=True)
    if encode_event == 'one-hot':
        train.drop('event', axis=1, inplace=True)
    elif encode_event == 'label':
        train.drop(['dislike', 'like', 'skip', 'view'], axis=1, inplace=True)
        enc = LabelEncoder()
        train['event'] = enc.fit_transform(train['event'])
        if verbose:
            print('like, view, skip, dislike = ', enc.transform(['like', 'view', 'skip', 'dislike']))
    else:
        train.drop(['dislike', 'like', 'skip', 'view'], axis=1, inplace=True)

    train.fillna(0., inplace=True)
    test.fillna(0., inplace=True)
    test.drop(['like', 'dislike', 'view', 'skip'], axis=1, inplace=True)

    return train, test
