import os
import numpy as np
def main():
    env_list = ['heist', 'chaser']
    file_obj = open("/home/kbc/Time-DA/result", "w")

    for env in env_list:
        for exp in ['drac_rccj_crop', 'ucb_drac', 'ucb_drac_coef_1.5', 'ucb_drac_coef_2']:
            for opt in ['test_bg', 'test_lv', 'train']:
                file_obj.write(env + '_easybg_' + exp + '__' + opt + '.txt')
                file_obj.write("\n")
                sum = 0.0
                tr = 0.0
                score_list = []
                for t in ['try1', 'try2', 'try3', 'try4', 'try5']:
                    
                    file_name = t + '_' + env + '_easybg_' + exp + '__' + opt + '.txt'
                    if os.path.isfile('/home/kbc/Time-DA/test_log/' + file_name):
                        read_file = open('/home/kbc/Time-DA/test_log/' + file_name, "r")
                        score = read_file.read()
                        score_list.append(float(score))
                        sum += float(score)
                        tr += 1.0
                        file_obj.write(file_name)
                        file_obj.write("\n")
                        file_obj.write(score)
                        file_obj.write("\n")
                file_obj.write('avg')
                file_obj.write("\n")
                if tr:
                    file_obj.write(str(sum / tr))
                else:
                    file_obj.write("None")
                file_obj.write("\n")
                print(score_list)
                file_obj.write(str(np.std(score_list, dtype = np.float32)))
                file_obj.write("\n")
if __name__ == '__main__':
    main()
