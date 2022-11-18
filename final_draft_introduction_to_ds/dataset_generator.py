import pandas as pd


## Путь до файлов датасета
path_to_file_1 = 'data/ga_sessions.csv'
path_to_file_2 = 'data/ga_hits.csv'


def generate_dataset():
    def create_dataset_key_action(df_sessions, df_hits):

        def target_action(df):
            if df['event_action'] == 'sub_car_claim_click':
                return 1
            elif df['event_action'] == 'sub_car_claim_submit_click':
                return 1
            elif df['event_action'] == 'sub_open_dialog_click':
                return 1
            elif df['event_action'] == 'sub_custom_question_submit_click':
                return 1
            elif df['event_action'] == 'sub_call_number_click':
                return 1
            elif df['event_action'] == 'sub_callback_submit_click':
                return 1
            elif df['event_action'] == 'sub_submit_success':
                return 1
            elif df['event_action'] == 'sub_car_request_submit_click':
                return 1
            else:
                return 0

        df_hits['y'] = df_hits.apply(target_action,
                                     axis=1)  # отмечаем на датасете действия, в которых клиенты совершили target
        df_1 = df_hits[df_hits['y'] == 1].copy()  # отделим действия target от остальных

        # удалим дубликаты сессий target, так как если клиент совершил несколько target, то они приравниваются к одному target

        df_1 = df_1.drop_duplicates(subset=['session_id'])
        df_1 = df_1[['session_id', 'y']]  # оставим только колонку y
        df_sessions = df_sessions.set_index('session_id')
        df_1 = df_1.set_index('session_id')
        df_sessions['y'] = df_1['y']  # target сессии объединяются с df_sessions по индексу
        df_sessions.y = df_sessions.y.astype("str")
        df_sessions = df_sessions.replace(
            {'y': {'nan': '0.0'}})  # заменим nan на 0.0 - клиент не совершил целевое действие
        df_sessions.y = df_sessions.y.astype("float").astype("int")

        # разделим df_sessions датасет на два датасета - сессии где клиент совершил целевое target и не совершил

        df_sessions_1 = df_sessions[df_sessions.y == 1]
        df_sessions_0 = df_sessions[df_sessions.y == 0]

        # удалим сессии в обоих датасетах по client_id, оставим в датасетах только информацию совершил клиент target или нет
        # причем в df_sessions_1 нам не важно какие сессии удалять, а в df_sessions_0 останется последняя сессия target = 0

        df_sessions_1 = df_sessions_1.drop_duplicates(subset=['client_id'])
        df_sessions_0 = df_sessions_0.drop_duplicates(subset=['client_id'], keep='last')
        df_sessions_y = df_sessions_1.append(df_sessions_0)  # объединим датасеты, вверху будут target = 1

        # теперь клиенты target = 1 стоят в начате датафрейма и при удалении дубликатов сессий они сохранятся (keep='first')
        # в df_sessions_y сохранились дубликаты так как в target=0 сохранились client_id да того как они совершили target = 1

        df_sessions_y = df_sessions_y.drop_duplicates(subset=['client_id'], keep='first')

        # удаление колонок, не участвующих в модели по условиям задачи

        df_key_action = df_sessions_y.set_index('client_id')
        df_key_action = df_key_action.drop(
            columns=['visit_date', 'visit_time', 'visit_number'])
        df_key_action = df_key_action.sample(frac=1)  # перемешать

        return df_key_action.copy()

    df_1 = pd.read_csv(path_to_file_1)
    df_2 = pd.read_csv(path_to_file_2)
    df_result = create_dataset_key_action(df_1, df_2)

    return df_result.to_csv('data/df_key_action.csv')


if __name__ == '__main__':
    generate_dataset()