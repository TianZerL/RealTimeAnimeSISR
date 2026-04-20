# RealTimeAnimeSISR
Training and experimental code for real-time anime super-resolution.

# Install
```Shell
git clone https://github.com/TianZerL/RealTimeAnimeSISR
cd RealTimeAnimeSISR
# You may want to install `torch` and `torchvision` by yourself to use GPU.
pip install -r requirements.txt
```

# Usage
```Shell
# Before training or testing, remember to set the dataset in the yml file.
# Train
python train.py -opt options/arnet_cmp_expr/arnet_best_net_f8_b8_rrrb_plain_gray_ma1080v3_bicubic_option.yml --auto_resume
# Test
python test.py -opt options/arnet_cmp_expr/test/arnet_best_net_f8_b8_rrrb_plain_gray_ma1080v3_bicubic_test_option.yml
```
