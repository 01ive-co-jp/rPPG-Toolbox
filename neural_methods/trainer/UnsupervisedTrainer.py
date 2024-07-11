import os
import pickle

from neural_methods.trainer.BaseTrainer import BaseTrainer


class UnsupervisedTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()

    def save_test_outputs(self, predictions, labels, config, method_name):
        output_dir = config.UNSUPERVISED.OUTPUT_SAVE_DIR if hasattr(config.UNSUPERVISED, 'OUTPUT_SAVE_DIR') else ''
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Filename ID to be used in any output files that get saved
        if config.TOOLBOX_MODE == 'unsupervised_method':
            filename_id = f"{method_name}_{config.UNSUPERVISED.DATA.DATASET}"
        else:
            raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')
        output_path = os.path.join(output_dir, filename_id + '_outputs.pickle')

        data = dict()
        data['predictions'] = predictions
        data['labels'] = labels
        data['label_type'] = config.UNSUPERVISED.DATA.PREPROCESS.LABEL_TYPE
        data['fs'] = config.UNSUPERVISED.DATA.FS

        with open(output_path, 'wb') as handle: # save out frame dict pickle file
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('Saving outputs to:', output_path)