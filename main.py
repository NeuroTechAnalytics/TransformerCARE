
from collections import defaultdict
from data import load_data, segment_audios
from config import *
from utils.global_constants import *
from train import Trainer


train_data_path = 'dataset/train_data_2021'
test_data_path = 'dataset/test_data_2021'

segmented_train_data_path = 'segmentation/train_data'
segmented_test_data_path = 'segmentation/test_data'

# result_path = 'results/run_1'
# results_sheet = 'main_results.xlsx'
# eval_probs_sheet = 'predicted_probability.xlsx'


if __name__ == '__main__':

    dataset = defaultdict(lambda: {})

    dataset[TRN][SUB], train_label_encoder = load_data(train_data_path)
    dataset[VAL][SUB], val_label_encoder = load_data(test_data_path)
    dataset[TST] = dataset[VAL].copy()

    segment_audios(dataset[TRN][SUB].path.values, segmented_train_data_path, segment_size, 5, overlap)
    segment_audios(dataset[VAL][SUB].path.values, segmented_test_data_path, segment_size, 5, overlap)

    dataset[TRN][SEG], _ = load_data(segmented_train_data_path)
    dataset[VAL][SEG], _ = load_data(segmented_test_data_path)
    dataset[TST] = dataset[VAL].copy()   

    trainer = Trainer(dataset)
    trainer.train_and_evaluate()

    final_results = trainer.reports.final_results
    trainer.testing(VOTING, str(final_results[VOTING]['best_epoch'])+'.pt', True)
    trainer.testing(SEG, str(final_results[EMBED_BASED]['best_epoch'])+'.pt', True )
    final_results, evaluation_results, evaluation_probs = trainer.testing(EMBED_BASED, best_embed_model_path, True)



    
        