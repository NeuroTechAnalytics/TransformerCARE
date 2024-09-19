DISTIL_HUBERT = 'ntu-spml/distilhubert'
HUBERT = 'facebook/hubert-base-ls960'
WAV2VEC2 = 'facebook/wav2vec2-base-960h'

seed = 10
lr = [1.5e-5, 1e-3]
epochs = [40, 200]
bs = [2, 8]
segment_size = 14
overlap = 0.25
transformer_checkpoint = HUBERT
segnet_extended = True
segnet_midsize = 256
segnet_dropout = 0
num_labels = 2
subnet_extended = True
subnet_insize = 768
subnet_midsize = 128
subnet_dropout = 2

best_segment_model_path = 'best_segment_model.pt'
best_score_model_path = 'best_score_model.pt'
best_embed_model_path = 'best_embed_model.pt'