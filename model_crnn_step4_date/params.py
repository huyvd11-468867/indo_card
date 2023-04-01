alphabet = "0123456789-*"

keep_ratio = False # whether to keep ratio for image resize
manualSeed = 1234 # reproduce experiment
random_sample = True # whether to sample the dataset with random sampler
imgH = 32 # the height of the input image to network
imgW = 100 # the width of the input image to network
nh = 256 # size of the lstm hidden state
nc = 1
pretrained = '' # path to pretrained model (to continue training)
expr_dir = 'expr' # where to store samples and models
dealwith_lossnan = False # whether to replace all nan/inf in gradients to zero

nclass = 1
