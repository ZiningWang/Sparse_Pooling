import os
import numpy as np

def main():
    write_path = 'avod/data/outputs/pyramid_people_example_train/predictions/kitti_predictions_3d/val/0.1/120000/data/'
    write_name = 'pedestrian'#'cyclist'#'pedestrian'
    result_type = 'ground'#'3d'
    PR_file = os.path.join(write_path,('../plot/'+write_name+'_detection_'+result_type+'.txt'))
    #try:
    PRs = np.loadtxt(PR_file)
    print('file loaded')
    APs = np.sum(PRs[0:-1,1:4]*(PRs[1:,0:1]-PRs[0:-1,0:1]),axis=0)
    conclusion_path = os.path.join(write_path,'../../../conclusion.txt')
    with open(conclusion_path,'a+') as conclusion_file:
        print('conclusion filed opened')
        conclusion_file.write('iteration '+': ')
        conclusion_file.write('\nrecall            :\n')
        PRs[:,0].tofile(conclusion_file," ",format='%.3f')
        conclusion_file.write('\nprec_easy, AP: %.2f :\n'%APs[0])
        PRs[:,1].tofile(conclusion_file," ",format='%.3f')
        conclusion_file.write('\nprec_mod , AP: %.2f :\n'%APs[1])
        PRs[:,2].tofile(conclusion_file," ",format='%.3f')
        conclusion_file.write('\nprec_hard, AP: %.2f :\n'%APs[2])
        PRs[:,3].tofile(conclusion_file," ",format='%.3f')
    print('APs: %.3f, %.3f, %.3f'%(APs[0],APs[1],APs[2]))
    #except:
    #    #f_log.write('No object detected')
    #    print('No object detected')
if __name__ == "__main__":
    main()
