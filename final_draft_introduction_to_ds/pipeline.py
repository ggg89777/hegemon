import dill as dill


def pipeline_processing():

    import datetime
    import pandas as pd
    import itertools

    from sklearn.preprocessing import StandardScaler, FunctionTransformer
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier

    from scipy.stats import entropy
    from sklearn.pipeline import Pipeline

    import warnings
    warnings.filterwarnings("ignore")
    from typing import List, NewType
    EntropyCategoricalEmbedderObject = NewType('EntropyCategoricalEmbedderObject',
                                               object)  # creating new type to annotate in our class

    def missing_values(df_old):
        missing_values = ((df_old.isna().sum() / len(df_old)) * 100).sort_values(ascending=False)
        missing_values = pd.DataFrame(missing_values)  # датафрейм процент пропущенных значений для каждого признака
        df_missing_drop = missing_values[missing_values[0] > 80]  # датафрейм признаков которые подлежат удалению
        df_missing_mode = missing_values[
            (missing_values[0] < 20) & (
            (missing_values[0] > 0))]  # датафрейм признаков которые подлежат заполнению модой
        df_missing_other = missing_values[
            (missing_values[0] > 20) & ((missing_values[0] < 80))]  # датафрейм признаков кот.
        # подлежат заполению other
        missing_drop = df_missing_drop.index.to_list()
        missing_mode = df_missing_mode.index.to_list()
        missing_other = df_missing_other.index.to_list()

        # списки колонок с пропущенными значениями
        df_old = df_old.drop(columns=missing_drop)
        df_old = df_old.drop(columns='client_id')

        for column in missing_mode:
            df_old[column] = df_old[column].fillna(df_old[column].mode()[0])

        for column in missing_other:
            df_old[column] = df_old[column].fillna('other')

        return df_old.copy()

    # в датасете имеются скрытые пропуски. так как их незначительное количество для упрощения заполним их other

    def hidden_gaps(df_gaps):
        list_columns = df_gaps.columns
        for column in list_columns:
            df_gaps.loc[df_gaps[column] == '(not set)', column] = 'other'
            df_gaps.loc[df_gaps[column] == '(none)', column] = 'other'

        return df_gaps.copy()

    def normal_device_browser(df_db):

        def normal_str(str_list):
            if str_list[0:9] == 'Instagram':
                str_list = 'Instagram'
            if str_list[0:7] == 'Threads':
                str_list = 'Threads'
            if str_list[0:5] == '[FBAN':
                str_list = 'Facebook'
            if str_list[0:10] == 'helloworld':
                str_list = 'other'

            return str_list

        df_db.device_browser = df_db.device_browser.apply(normal_str)

        return df_db.copy()

    def normal_screen_resolution(df_sr):

        def pixels(screen_resolution):
            screen_list = screen_resolution.split('x')
            pixels = int(screen_list[0]) * int(screen_list[1])

            return pixels

        def pixels_range(pixel):
            if pixel == 0:
                resolution = 'other'
            elif (pixel > 0) & (pixel < 300000):
                resolution = 'low'
            elif pixel > 375000:
                resolution = 'high'
            else:
                resolution = 'medium'

            return resolution

        df_sr.loc[df_sr['device_screen_resolution'] == '(not set)', 'device_screen_resolution'] = '0x0'
        df_sr.device_screen_resolution = df_sr.device_screen_resolution.apply(pixels)
        df_sr.device_screen_resolution = df_sr.device_screen_resolution.apply(pixels_range)

        return df_sr.copy()

    class EntropyCategoricalEmbedder:
        """Unsupervised categorical embedder based on group counts and entropy calculation

        fit - get dictionary for the transformation of categorical objects into embeddings
        transform - map the dictionary onto your categorical dataset to get the embeddings
        """

        def __init__(self):
            self.substitute_dict = {}  # resulting dictionary to transform the objects into embs

        def __repr__(self):
            return self.__class__.__name__ + "()"

        @staticmethod
        def cat_prep(data: pd.DataFrame) -> pd.DataFrame:

            """change category names for simplification

            format -> category-name_category-name
            """

            data_new = data.copy()
            for col in data.columns:
                data_new[col] = data[col].apply(lambda x: col + '_' + str(x))
            return data_new

        def fit(self, df_train: pd.DataFrame,
                verbose: bool = True) -> EntropyCategoricalEmbedderObject:  # we created this custom type earlier
            """Create dictionary to map on the dataset

            !!!Works only with categorical datasets!!!
            dataset - pandas DataFrame with only categorical columns in str format (after cat_prep)
            (each row is our object to get an embedding for)
            """

            feature_list = list(df_train.columns)
            df = df_train.copy()
            df['id'] = df.index
            for group_key in feature_list:
                passive_keys = feature_list[:]
                passive_keys.remove(group_key)

                category_embedding_mapping = {}
                for passive_key in passive_keys:
                    if verbose:
                        print('--- groupby: group_key - ', group_key, '### passive_key - ', passive_key, '---')
                    group = df.groupby([group_key, passive_key])['id'].count()
                    group = group.unstack().fillna(0)
                    entropy_values = group.apply(entropy, axis=1)
                    for cat, entropy_value in entropy_values.to_dict().items():
                        if cat in category_embedding_mapping:
                            category_embedding_mapping[cat].extend([entropy_value])
                        else:
                            category_embedding_mapping[cat] = [entropy_value]

                self.substitute_dict[group_key] = category_embedding_mapping
            return self

        def transform(self, dataset: pd.DataFrame,
                      fill_unknown_cat_value: int = 0,
                      verbose: bool = False) -> List[list]:
            """Get embedding for each categorical row of the dataset

            !!!Works only with categorical datasets!!!
            dataset - pandas DataFrame with only categorical columns in str format (after cat_prep)
            (each row is our object to get an embedding for)
            fill_unknown_cat_value - the value to fill embedding vector for unknown categories
            """

            dataset = dataset.copy()
            feature_list = list(dataset.columns)
            emb_size = len(feature_list) - 1
            if verbose:
                print("Mapping vectors to categories...")
            for f in feature_list:
                dataset[f] = dataset[f].map(self.substitute_dict[f])
                dataset[f] = dataset[f].fillna('empty')
                dataset[f] = dataset[f].apply(lambda x: [fill_unknown_cat_value] * emb_size if x == 'empty' else x)

            embeddings_list = []
            if verbose:
                print("Creating an embedding for each row...")
            for row in dataset[feature_list].itertuples():
                embeddings_list.append(list(itertools.chain(*row[1:])))

            return embeddings_list

    def encoder_cat(df_enc):
        df_enc = EntropyCategoricalEmbedder.cat_prep(df_enc)
        embedder = EntropyCategoricalEmbedder()
        embedder.fit(df_enc, verbose=False)
        df_feat = embedder.transform(df_enc)

        return df_feat.copy()

    df = pd.read_csv('data/df_key_action.csv')

    X = df.drop(['y'], axis=1)
    y = df['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    preprocessor = Pipeline(steps=[
        ('normal_device_browser', FunctionTransformer(normal_device_browser)),
        ('normal_screen_resolution', FunctionTransformer(normal_screen_resolution)),
        ('remove_hidden_spaces', FunctionTransformer(hidden_gaps)),
        ('fill_missing_values', FunctionTransformer(missing_values)),
        ('column_transformer', FunctionTransformer(encoder_cat)),
        ('scaler', StandardScaler())
    ])

    perp = MLPClassifier()

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', perp)
    ])

    pipe.fit(X_train, y_train)
    probs = pipe.predict_proba(X_test)
    score = roc_auc_score(y_test, probs[:, 1])
    print(f'model: {type(perp).__name__}, roc_auc_score: {score:.4f}')

    pipe.fit(X, y)
    print("Модель обучена на всей выборке")

    return {
        "model": pipe,
        "metadata": {
            "name": "Конвейер предсказания целевого действия клиентов",
            "author": "Алексей К",
            "version": 1,
            "date": datetime.datetime.now(),
            "type": type(pipe.named_steps["classifier"]).__name__,
            "roc_auc_score": score
        }
    }


def dill_dump():
    with open("model/sber_auto_pipe.pkl", 'wb') as file:
        dill.dump(pipeline_processing(), file)


if __name__ == '__main__':
    dill_dump()