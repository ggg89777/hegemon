import json
import os
import dill
import pandas as pd


path = os.environ.get('PROJECT_PATH', '.')
model_filename = f'{path}/data/models/cars_pipe_202209261253.pkl'
json_filename = f'{path}/data/test/7310993818.json'
json_dir = f'{path}/data/test'


def predict():

    def pred(filename):
        with open(model_filename, 'rb') as file:
            model = dill.load(file)
        with open(filename, 'r') as file:
            json_df = json.load(file)
        df = pd.DataFrame(json_df, index = [0])
        pred = model.predict(df)
        pred = pd.DataFrame(pred, index = df['id'])
        pred = pred.rename(columns = {0: "pred"})
        pred.index.names = ['car_id']

        return pred


    def json_path_func(dir):
        json_list = os.listdir(dir)
        json_path = []
        for list_json in json_list:
            json_path.append(f'{path}/data/test/{list_json}')

        return json_path


    list_for_pred = json_path_func(json_dir)
    df = pred(json_filename)
    for filename in list_for_pred:
        df_json = pred(filename)
        df = pd.concat([df, df_json], axis=0)
    df = df.iloc[1:]
    df.to_csv(f'{path}/data/predictions/predict.csv')


if __name__ == '__main__':
    predict()