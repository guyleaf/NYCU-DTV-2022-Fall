$dataDir = "data/demo/images"
# $dataDir = "data"
$datasetName = "DEMO"
# $datasetName = "MOT20-ALL"
$framerate = 30

$suffix = "mot20_crowdhuman_deformable_multi_frame"
$model = "models/$suffix/checkpoint_epoch_50.pth"

$baseOutputDir = "F:/NYCU/DTV/hw3"

for ($i = 5; $i -le 50; $i += 5)
{
    $fileName = "demo_reid_${suffix}_${i}" 
    $outputDir = "$baseOutputDir/$fileName"

    python src/track.py with reid `
        obj_detect_checkpoint_file=$model `
        dataset_name=$datasetName `
        data_root_dir=$dataDir `
        output_dir=$outputDir `
        write_images=pretty `
        tracker_cfg.inactive_patience=$i

    wsl ffmpeg -framerate $framerate `
        -i "$outputDir/**/%06d.png" `
        -vcodec "libx264" "$baseOutputDir/videos/$fileName.mp4"
}
