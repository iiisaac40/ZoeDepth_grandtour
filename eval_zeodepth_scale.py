import os
import pandas as pd

SCHEDULE = True
submit_dir ="/cluster/home/haozhu1/Thesis/result/.eval_zoedepth"
if not os.path.exists(submit_dir):
    os.makedirs(submit_dir)

eval_list = ['/home/output/GrandTour/2024-11-02-17-10-25', '/home/output/GrandTour/2024-11-14-11-17-02'] #, '/home/output/GrandTour/2024-11-15-11-18-14']

mode = 'zoedepth'
pretrained_ckpt = 'url::https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_K.pt'

max_depth = 60

txt_file = '/mnt/txt_files/'
for depth_alignment in ['FALSE', 'TRUE']:
    metric_summary = {}
    csv_pattern = 'ZoeDepth_noAlignment_metric' if depth_alignment == 'FALSE' else 'ZoeDepth_Alignment_metric'


    for accum_frames in [1, 5, 25, 50, 100, 150]:

        test_txt = os.path.join(txt_file, f'test_{str(accum_frames)}_files.txt')
        csv_file = test_txt.replace('files.txt', f'{csv_pattern}.csv')

        print(f"evaluating dataset: accumulate level {str(accum_frames)}")
        
        content = f"""#!/bin/bash

#SBATCH --account=es_hutter
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --gres=gpumem:12288m
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=13312
#SBATCH --tmp=90000
#SBATCH --output="/cluster/home/haozhu1/Thesis/.out/zoedepth_eval_{str(accum_frames)}_depth{str(max_depth)}_out.log"
#SBATCH --error="/cluster/home/haozhu1/Thesis/.out/zoedepth_eval_{str(accum_frames)}_depth{str(max_depth)}_out.log"
#SBATCH --open-mode=truncate

mkdir -p $TMPDIR/GrandTour
tar -xf /cluster/scratch/haozhu1/Thesis/container/grandtour_depth_benchmark2.tar -C $TMPDIR
tar -xf /cluster/scratch/haozhu1/depth_data/updated_images/GrandTour/GrandTour.tar  -C $TMPDIR/GrandTour

# tar -xf /cluster/scratch/haozhu1/depth_data/eval_images/KITTI.tar -C $TMPDIR/GrandTour
# sed -i 's|/cluster/scratch/haozhu1/depth_data/eval_images|/mnt|g' $TMPDIR/GrandTour/KITTI/val_pairs.txt

module load stack/2024-04 gcc/8.5.0 cuda/12.1.1 eth_proxy

apptainer exec --nv --containall --writable --env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu   \
--env HF_HOME=/mnt/.cache/huggingface \
--env TRANSFORMERS_CACHE=/mnt/.cache/huggingface \
--env XDG_CACHE_HOME=/mnt/.cache \
--env MPLCONFIGDIR=/mnt/.config/matplotlib \
--bind $TMPDIR/GrandTour:/mnt/ \
$TMPDIR/grandtour_depth_benchmark2.sif /bin/bash -c " \
export HOME=/home && export KLEINKRAM_ACTIVE=ACTIVE && \
source /opt/conda/etc/profile.d/conda.sh && \
conda activate grandtour && \
source /opt/ros/noetic/setup.bash && \
source /home/grand_tour_depth_benchmark/catkin_ws/devel/setup.bash && \
cd /home/grand_tour_depth_benchmark/evaluation/ZoeDepth_grandtour && \
python -c 'import torch; print(torch.cuda.device_count())' && \
ls /mnt/ && \
python sanity_hub.py && \
python /home/grand_tour_depth_benchmark/evaluation/ZoeDepth_grandtour/evaluate.py \
                        -m zoedepth -d grandtour -p {pretrained_ckpt} \
                        --accumulate_level {accum_frames}  \
                        --csv_path {csv_file} --depth_alignment {depth_alignment}

"

cp -r $TMPDIR/GrandTour/*/*.csv /cluster/scratch/haozhu1/depth_data/updated_images
cp -r $TMPDIR/GrandTour/GrandTour/visualizations /cluster/scratch/haozhu1/depth_data/updated_images
 

exit 0
                            """

# bash /home/grand_tour_depth_benchmark/third_parties/Marigold_grandtour/script/eval/61_infer_grandtour.sh '' '' '/mnt' '/mnt' 
        script_path = os.path.join(submit_dir, f"eval_zoedepth_accum{accum_frames}.sh")
        with open(script_path, "w") as file:
            file.write(content)
    
        if SCHEDULE:
            os.system(f"sbatch {script_path}")
            
            
