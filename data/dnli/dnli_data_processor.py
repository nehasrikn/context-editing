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
class DeltaNLIExample:
    example_id: str
    premise: str
    edited_hypothesis: Optional[str]
    hypothesis: str
    update: str
    update_type: str
    label: int
    edited_label: Optional[int]

    def get_partial_input(self):
        return {'sentence1': self.update, 'label': self.label}

    def get_full_input(self):
        return {
            'sentence1': f'{self.premise} {self.hypothesis}', 
            'sentence2': f'{self.update}',
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

class DeltaNLIDataProcessor:
    """
    Data processing class for DeltaNLI data. Takes in a directory for a 
    defeasible data source (defeasible-snli, defeasible-atomic, defeasible-social).
    """

    def __init__(self, directory_path: str = 'raw_data/defeasible-snli'):
        self.directory_path = directory_path

        self.all_data = {
            'train': self.load_dataset('train'),
            'dev': self.load_dataset('dev'),
            'test':  self.load_dataset('test')
        }

    def get_split_data(self, split: str) -> List[DeltaNLIExample]:
        return self.all_data[split]

    def load_dataset(self, split: str) -> List[DeltaNLIExample]:
        """
        Reads in data from raw jsonl files.
        """
        data = []
        fname = '%s/%s.jsonl' % (self.directory_path, split)

        for i, json_str in enumerate(list(open(fname, 'r'))):
            result = json.loads(json_str)

            if not all(v for v in [result['Hypothesis'], result['Update']]):
                continue

            data.append(
                DeltaNLIExample(
                    example_id='%s.%d' % ('dnli', i),
                    premise=result['Premise'] if 'social' not in self.directory_path else "", #social has no premises
                    hypothesis=result['Hypothesis'],
                    update=result['Update'],
                    update_type=result['UpdateType'],
                    label=0 if result['UpdateType'] == 'weakener' else 1,
                )
            )
        print('Loaded %d nonempty %s examples...' % (len(data), split))
        return data

    @staticmethod
    def write_processed_data(
        data: List[DeltaNLIExample],
        split: str,
        partial_input: bool,
        data_dir:str='processed_data' 
    ) -> None:

        input_type = 'partial_input' if partial_input else 'full_input'

        if not os.path.exists(os.path.join(data_dir, input_type)):
            os.makedirs(os.path.join(data_dir, input_type))

        fname = f'{data_dir}/{input_type}/dnli_{input_type}_{split}.csv'
        fieldnames = ['sentence1', 'label'] if partial_input else ['sentence1', 'sentence2', 'label']

        with open(fname, 'w', newline='\n') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for example in data:
                writer.writerow(example.get_partial_input() if partial_input else example.get_full_input())
                
def write_dnli_snli_data() -> None:
    dnli_snli = DeltaNLIDataProcessor(directory_path='raw_data/defeasible-snli')
    for split in ['train', 'dev', 'test']:
        DeltaNLIDataProcessor.write_processed_data(
            data=dnli_snli.get_split_data(split),
            split=split,
            partial_input=True,
            data_dir='processed_data/dnli_snli'
        )
        DeltaNLIDataProcessor.write_processed_data(
            data=dnli_snli.get_split_data(split),
            split=split,
            partial_input=False,
            data_dir='processed_data/dnli_snli'
        )

def write_dnli_data() -> None:
    dnli_snli = DeltaNLIDataProcessor(directory_path='raw_data/defeasible-snli')
    dnli_atomic = DeltaNLIDataProcessor(directory_path='raw_data/defeasible-atomic')
    dnli_social = DeltaNLIDataProcessor(directory_path='raw_data/defeasible-social')

    for split in ['train', 'dev', 'test']:
        split_data = dnli_snli.get_split_data(split) + dnli_atomic.get_split_data(split) + dnli_social.get_split_data(split)

        DeltaNLIDataProcessor.write_processed_data(
            data=split_data,
            split=split,
            partial_input=True,
            data_dir='processed_data/dnli'
        )
        DeltaNLIDataProcessor.write_processed_data(
            data=split_data,
            split=split,
            partial_input=False,
            data_dir='processed_data/dnli'
        )

        
if __name__ == '__main__':

    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')

    write_dnli_snli_data()
    write_dnli_data()


