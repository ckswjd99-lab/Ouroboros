import torchvision
import os

from ipconv.models.fasterrcnn_resnet50_fpn import FasterRCNN_ResNet50_FPN_Contexted

def main():
    
    model = FasterRCNN_ResNet50_FPN_Contexted("cuda")

    # sequence_names = os.listdir("/data/DAVIS/JPEGImages/480p")
    # gops = [1, 6, 30, 100]

    # sequence_names = ['bear', 'bus', 'camel', 'dog-gooses', 'loading', 'sheep', 'skate-park']
    sequence_names = ['bear']
    gops = [1, 6, 30, 100]

    for sequence_name in sorted(sequence_names):
        if os.path.exists(f"./output/contexted_inference/{sequence_name}"):
            continue

        # save result to file
        os.makedirs(f"./output/contexted_inference/{sequence_name}", exist_ok=True)
        log_file = open(f"./output/contexted_inference/{sequence_name}/log.txt", "w")
        log_file.write(f"Sequence: {sequence_name}\n")


        for gop in gops:
            print(f"Processing sequence: {sequence_name}, GOP: {gop}")
            avg_compute_rate, avg_iou_gt, avg_iou_full, inference_results = model.validate_DAVIS(sequence_name, gop)
            print(f"  - Average recompute rate: {avg_compute_rate}")
            print(f"  - Average IoU (GT): {avg_iou_gt}")
            print(f"  - Average IoU (full): {avg_iou_full}")
            print()

            log_file.write(f"  GOP: {gop}\n")
            log_file.write(f"    - Average recompute rate: {avg_compute_rate}\n")
            log_file.write(f"    - Average IoU (GT): {avg_iou_gt}\n")
            log_file.write(f"    - Average IoU (full): {avg_iou_full}\n")
            log_file.write("\n")

        

if __name__ == "__main__":
    main()
        