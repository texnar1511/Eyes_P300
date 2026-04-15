from experiment import Experiment
import json

with open('settings.json', encoding = 'utf-8') as file:
    data_config = json.load(file)

exp = Experiment(data_config)
exp.fit()