

#python JRA/preprocessed/utils/scripts/split-dataset.py --input-dataset C:/Users/Will6/OneDrive/Documents/MemorialUniversity/JetReversalAnalysis/JRA/preprocessed/data/stream_dataset_20.csv
#python JRA/preprocessed/utils/scripts/split-dataset.py --input-dataset C:/Users/Will6/OneDrive/Documents/MemorialUniversity/JetReversalAnalysis/JRA/preprocessed/data/reversals_dataset_20.csv
#python JRA/preprocessed/utils/scripts/split-dataset.py --input-dataset /mnt/abahari/reversals_dataset_neg2pos_10000.csv
#echo "+ Split stream, reversal pos2neg, and reversal neg2pos csv datasets"



##### STATS #####
### Reversals
#python JRA/preprocessed/utils/scripts/generate-windowed-dataset.py --input-dataset JRA/preprocessed/data/reversals_dataset_40-stats.csv --output-zarr JRA/preprocessed/data/reversals_dataset_40-stats.zarr --num-windows 54000 --num-timesteps 1000 --mode reversal --normalize n
#echo "+ Generated windows for pos2neg and neg2pos reversal stats datasets and saved to Zarr files"

### Stream
#python JRA/preprocessed/utils/scripts/generate-windowed-dataset.py --input-dataset JRA/preprocessed/data/stream_dataset_40-stats.csv --output-zarr JRA/preprocessed/data/stream_dataset_40-stats.zarr --num-windows 54000 --num-timesteps 50000 --mode stream --normalize n
#echo "+ Generated windows for stream stats dataset and saved to Zarr file"

### Combine
#python JRA/preprocessed/utils/scripts/merge-zarr-in-ram.py --datasets JRA/preprocessed/data/reversals_dataset_40-stats.zarr JRA/preprocessed/data/stream_dataset_40-stats.zarr --output-zarr JRA/preprocessed/data/dataset-stats.zarr
#echo "+ Merged stream and reversal stats Zarr datasets"

#python JRA/preprocessed/utils/scripts/get-stats-windowed.py --input-dataset JRA/preprocessed/data/dataset-stats.zarr --output-csv JRA/preprocessed/data/stats5050.csv
#echo "+ Created stats"
######################

##### TEST #####
# Used in inference for visualization
python JRA/preprocessed/utils/scripts/generate-windowed-dataset.py --input-dataset JRA/preprocessed/data/reversals_dataset_40-test.csv --output-zarr JRA/preprocessed/data/dataset-reversal-windowed-test.zarr --num-windows -1 --num-timesteps 1000 --mode visualization --normalize y
python JRA/preprocessed/utils/scripts/generate-full-time-series.py --input-dataset JRA/preprocessed/data/reversals_dataset_40-test.csv --output-zarr JRA/preprocessed/data/dataset-reversal-full-test.zarr --num-timesteps 1000
#echo "+ Generated windowed and full time-series reversal test dataset"

#python JRA/preprocessed/utils/scripts/generate-windowed-dataset.py --input-dataset JRA/preprocessed/data/stream_dataset_40-test.csv --output-zarr JRA/preprocessed/data/dataset-stream-windowed-test.zarr --num-windows -1 --num-timesteps 50000 --mode visualization --normalize y
#python JRA/preprocessed/utils/scripts/generate-full-time-series.py --input-dataset JRA/preprocessed/data/stream_dataset_40-test.csv --output-zarr JRA/preprocessed/data/dataset-stream-full-test.zarr --num-timesteps 50000
#echo "+ Generated windowed and full time-series stream test dataset"
################