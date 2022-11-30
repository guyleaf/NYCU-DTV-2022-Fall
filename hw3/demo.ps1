$dataDir = "data/demo/images"

python src/track.py with dataset_name=DEMO data_root_dir=$dataDir output_dir=results/demo_deformable write_images=pretty
# python src/track.py with dataset_name=DEMO data_root_dir=$dataDir output_dir=results/demo write_images=pretty obj_detect_checkpoint_file=models/mots20_train_masks/checkpoint.pth generate_attention_maps=True
