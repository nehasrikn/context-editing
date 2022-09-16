import pandas
import csv
import json
import os
import numpy as np
import random
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Optional, Dict

RawModelPrediction = namedtuple('RawModelPrediction', 'score, prediction')

@dataclass
class SNLIExample:
    example_id: str
    premise: str
    edited_premise: Optional[str]
    hypothesis: str
    label_type: str
    label: int
    edited_label_type: Optional[str]
    edited_label: Optional[int]

    def get_partial_input(self):
        return {'sentence1': self.hypothesis, 'label': self.label}

    def get_full_input(self):
        return {
            'sentence1': self.premise, 
            'sentence2': self.hypothesis,
            'label': self.label
        }

    def set_partial_input_prediction(self, model_prediction: np.ndarray) -> None:
        self.partial_input_prediction = RawModelPrediction(
            score=model_prediction,
            prediction=np.argmax(model_prediction)
        )
    def set_full_input_prediction(self, model_prediction: np.ndarray) -> None:
        self.full_input_prediction = RawModelPrediction(
            score=model_prediction,
            prediction=np.argmax(model_prediction)
        )
    def set_scaled_full_input_prediction(self, model_prediction: np.ndarray) -> None:
        self.scaled_full_input_prediction = RawModelPrediction(
            score=model_prediction,
            prediction=np.argmax(model_prediction)
        )
    def set_scaled_partial_input_prediction(self, model_prediction: np.ndarray) -> None:
        self.scaled_partial_input_prediction = RawModelPrediction(
            score=model_prediction,
            prediction=np.argmax(model_prediction)
        )

class SNLIDataProcessor:

    SNLI_LABELS = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

    def __init__(self, directory_path: str = 'raw_data/snli_1.0'):
        self.directory_path = directory_path

        self.all_data = {
            'train': self.load_dataset('train'),
            'dev': self.load_dataset('dev'),
            'test':  self.load_dataset('test')
        }

    def get_split_data(self, split: str) -> List[SNLIExample]:
        return self.all_data[split]

    def load_dataset(self, split: str) -> List[SNLIExample]:
        data = []
        for i, json_str in enumerate(list(open('%s/snli_1.0_%s.jsonl' % (self.directory_path, split), 'r'))):
            result = json.loads(json_str)

            if result['gold_label'] == '-':
                continue

            data.append(
                SNLIExample(
                    example_id='%s.%d' % ('snli', i),
                    premise=result['sentence1'],
                    hypothesis=result['sentence2'],
                    label_type=result['gold_label'],
                    label=SNLIDataProcessor.SNLI_LABELS[result['gold_label']]
                )
            )
        print('Loaded %d nonempty %s examples...' % (len(data), split))
        return data

    @staticmethod
    def write_processed_data(
        data: List[SNLIExample],
        split: str,
        partial_input: bool,
        data_dir:str='processed_data' 
    ) -> None:

        input_type = 'partial_input' if partial_input else 'full_input'

        if not os.path.exists(os.path.join(data_dir, input_type)):
            os.makedirs(os.path.join(data_dir, input_type))

        fname = f'{data_dir}/{input_type}/snli_{input_type}_{split}.csv'
        fieldnames = ['sentence1', 'label'] if partial_input else ['sentence1', 'sentence2', 'label']

        with open(fname, 'w', newline='\n') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for example in data:
                writer.writerow(example.get_partial_input() if partial_input else example.get_full_input())
                

def write_snli_data() -> None:
    snli = SNLIDataProcessor(directory_path='raw_data/snli_1.0')

    for split in ['train', 'dev', 'test']:
        split_data = snli.get_split_data(split) 

        SNLIDataProcessor.write_processed_data(
            data=split_data,
            split=split,
            partial_input=True,
            data_dir='processed_data/snli'
        )
        SNLIDataProcessor.write_processed_data(
            data=split_data,
            split=split,
            partial_input=False,
            data_dir='processed_data/snli'
        )


if __name__ == '__main__':

    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')
    
    write_snli_data()

