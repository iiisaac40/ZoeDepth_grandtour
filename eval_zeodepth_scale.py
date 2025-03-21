import os
import pandas as pd

eval_list = ['/home/output/GrandTour/2024-11-02-17-10-25'] #, '/home/output/GrandTour/2024-11-14-11-17-02', '/home/output/GrandTour/2024-11-15-11-18-14']
depth_alignment = 'TRUE'

mode = 'zoedepth'
pretrained_ckpt = 'url::https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_K.pt'

max_depth = 60

for depth_alignment in ['TRUE', 'FALSE']:
    metric_summary = {}
    csv_pattern = 'ZoeDepth_noAlignment_metric' if depth_alignment == 'FALSE' else 'ZoeDepth_Alignment_metric'


    for data_eval in eval_list:
        date_str = os.path.basename(data_eval)
        metric_summary[date_str] = {}

        for accumu_level in [1, 5, 25, 50, 100, 150]:
            txt_file = os.path.join(data_eval, f"accumulate_{str(accumu_level)}_pairs.txt")
            csv_path = txt_file.replace('_pairs.txt', f'_2024-11-02-17-10-25_{csv_pattern}.csv')
            print(f"evaluating dataset: {data_eval} in accumulate level {str(accumu_level)}")
            os.system(f"python /home/grand_tour_depth_benchmark/evaluation/ZoeDepth_grandtour/evaluate.py \
                        -m {mode} -d grandtour -p {pretrained_ckpt} \
                        --mission {date_str} --accumulate_level {str(accumu_level)} \
                        --csv_path {csv_path} --depth_alignment {depth_alignment}")
            
            
            df = pd.read_csv(csv_path)
            metrics_dict = df.iloc[0].to_dict()

            metric_summary[date_str][accumu_level] = metrics_dict
            
    rows = []
    for date, accumu_data in metric_summary.items():
        for accumu_level, metrics in accumu_data.items():
            row = {
                'date': date,
                'accumulation_level': accumu_level,
                **metrics  # Unpack metrics into columns
            }
            rows.append(row)

    summary_df = pd.DataFrame(rows)

    # Display as formatted table
    # =================================================================
    # Reorder columns (optional)
    cols = ['date', 'accumulation_level'] + [c for c in summary_df.columns if c not in ['date', 'accumulation_level']]
    summary_df = summary_df[cols]

    summary_df.to_csv(f"/home/output/2024-11-02-17-10-25_{csv_pattern}.csv", index=False)
            


