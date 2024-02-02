class CFG:
    debug = False
    batch_size = 13
    sequence_length = 32
    num_classes = 6
    num_feats = 3
    lr = 1e-4
    min_lr = 1e-5
    epochs = 500
    print_freq = 100
    resume = False

    num_classes_fine_tuning = 2
    classes_fine_tuning = ["Wave",   "ThumbsUp"]

    model_type = "AAGCN"

    add_feats = False
    add_phi = False

    add_joints1 = True     #Abl
    add_joints2 = True     #Abl    
    add_joints_mode = "ori"
    sam = True             #Abl
    only_dist = False       #Abl

    experiment_name = f"WaveThumbsUp_{model_type}_seqlen{sequence_length}_batch{batch_size}_finetuned_{'SAM_' if sam else ''}{'joints1_' if add_joints1 else ''}{'joints2_' if add_joints2 else ''}{'dist' if only_dist else ''}"

    plot_weights = True
    
    if add_feats:
        num_feats = 6
    
    classes = ["Grasp",   "Move",    "Negative",    "Position",    "Reach",   "Release"]
