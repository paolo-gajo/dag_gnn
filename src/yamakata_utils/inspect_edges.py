import os
from icecream import ic

diff_count = 0
adj_count = 0
firstsecond_count = 0
onediff_count = 0

for root, dirs, files in os.walk('/home/pgajo/Multitask-RFG/data/yamakata'):
    for F in files:
        if '.flow' in F:
            txt_lines = [el.strip() for el in open(os.path.join(root, F)).readlines()]
            for line in txt_lines:
                line_list = line.split()
                src_step = int(line_list[0])
                tgt_step = int(line_list[4])
                if src_step != tgt_step:
                    diff_count += 1
                if abs(src_step - tgt_step) == 1:
                    adj_count += 1
                if src_step == 1 and tgt_step == 2:
                    firstsecond_count += 1
                if src_step == 1 and tgt_step != src_step:
                    onediff_count += 1
                
                
ic(firstsecond_count)
ic(onediff_count)
ic(diff_count)
ic(adj_count)
