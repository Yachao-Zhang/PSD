python -B main_S3DIS_weak.py --gpu 0 --mode train --test_area 5 --labeled_point 1% --log_dir PSD_Area-5
python -B main_S3DIS_test.py --gpu 0 --mode test --test_area 5 --labeled_point 1% --log_dir PSD_Area-5
