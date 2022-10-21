# fsanomaly_rtfm

To train:
  1- Change root_path in the config 'configs/train_model.yaml' to the directory of the features containing both abnormal and normal feature folders
  2- Run "python training.py --config configs/train_model.yaml"

To test on Laplacian regularized VI model:
  1- Change root_path in the config 'configs/test_model_lap.yaml' to the directory of the features containing both abnormal and normal test feature folders
  2- Run "python training.py --config configs/test_model_lap.yaml"
  
To test on VI model: 
  1- Change root_path in the config 'configs/test_model.yaml' to the directory of the features containing both abnormal and normal test feature folders
  2- Run "python training.py --config configs/test_model.yaml"
