import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from data_management.read_csv import *
from data_management.functions import *
from visualization.visualize_frame import VisualizationPlot


###########################################
## read data
###########################################
def create_args():
    parser = argparse.ArgumentParser(description="ParameterOptimizer")
    # --- Input paths ---
    data_num = "01"
    # path = "D:/learn/23SS/guided research/highd-dataset-v1.0/highD-dataset/Python/data/"
    # print(os.getcwd())
    parser.add_argument(
        '--input_path',
        default="D:/learn/23SS/guided_research/highd-dataset-v1.0/data/" +
        data_num + "_tracks.csv",
        type=str,
        help='CSV file of the tracks')
    parser.add_argument(
        '--input_static_path',
        default="D:/learn/23SS/guided_research/highd-dataset-v1.0/data/" +
        data_num + "_tracksMeta.csv",
        type=str,
        help='Static meta data file for each track')
    parser.add_argument(
        '--input_meta_path',
        default="D:/learn/23SS/guided_research/highd-dataset-v1.0/data/" +
        data_num + "_recordingMeta.csv",
        type=str,
        help='Static meta data file for the whole video')
    parser.add_argument(
        '--pickle_path',
        default="D:/learn/23SS/guided_research/highd-dataset-v1.0/data/" +
        data_num + ".pickle",
        type=str,
        help=
        'Converted pickle file that contains corresponding information of the "input_path" file'
    )
    # --- Settings ---
    parser.add_argument('--visualize',
                        default=True,
                        type=lambda x: (str(x).lower() == 'true'),
                        help='True if you want to visualize the data.')
    parser.add_argument(
        '--background_image',
        default="D:/learn/23SS/guided_research/highd-dataset-v1.0/data/" +
        data_num + "_highway.jpg",
        type=str,
        help='Optional: you can specify the correlating background image.')

    # --- Visualization settings ---
    parser.add_argument(
        '--plotBoundingBoxes',
        default=True,
        type=lambda x: (str(x).lower() == 'true'),
        help='Optional: decide whether to plot the bounding boxes or not.')
    parser.add_argument(
        '--plotDirectionTriangle',
        default=True,
        type=lambda x: (str(x).lower() == 'true'),
        help='Optional: decide whether to plot the direction triangle or not.')
    parser.add_argument(
        '--plotTextAnnotation',
        default=True,
        type=lambda x: (str(x).lower() == 'true'),
        help='Optional: decide whether to plot the text annotation or not.')
    parser.add_argument(
        '--plotTrackingLines',
        default=True,
        type=lambda x: (str(x).lower() == 'true'),
        help='Optional: decide whether to plot the tracking lines or not.')
    parser.add_argument(
        '--plotClass',
        default=True,
        type=lambda x: (str(x).lower() == 'true'),
        help=
        'Optional: decide whether to show the class in the text annotation.')
    parser.add_argument(
        '--plotVelocity',
        default=True,
        type=lambda x: (str(x).lower() == 'true'),
        help=
        'Optional: decide whether to show the class in the text annotation.')
    parser.add_argument(
        '--plotIDs',
        default=True,
        type=lambda x: (str(x).lower() == 'true'),
        help=
        'Optional: decide whether to show the class in the text annotation.')

    # --- I/O settings ---
    parser.add_argument('--save_as_pickle',
                        default=True,
                        type=lambda x: (str(x).lower() == 'true'),
                        help='Optional: you can save the tracks as pickle.')
    parsed_arguments = vars(parser.parse_args())
    return parsed_arguments


###########################################
## main function
###########################################
def main(tracks):
    np.set_printoptions(suppress=True)
    # print(len(tracks))

    ###########################################
    ## filter following pattern
    ###########################################
    vf_tracks = filter_vf_tracks(tracks)
    # print(tracks[0][BBOX][:,0])
    # print('number',len(vf_tracks))

    ###########################################
    ## extract features needed
    ###########################################
    Vel, Acc, Nu, Dhw, Pr_a, pairs = extract_features(vf_tracks, tracks)
    # V_gen, A_gen, Nu_gen, D_gen, Pr_x_a_gen, pairs_gen = extract_features_gen(
    #     vf_tracks, tracks)
    # print(len(Vel))

    ###########################################
    ## construct data_array exported to MATLAB
    ###########################################
    # P_V = "p_v"
    # P_X = "p_x"
    # P_L = 'p_l'
    # X = "x"

    # for i in range(len(pairs)):
    #   pair = pairs[i]
    #   pair_len = pair[X].size
    #   Data_array = np.concatenate((pair[X].reshape((pair_len,1)), pair[X_VELOCITY].reshape((pair_len,1)), pair[X_ACCELERATION].reshape((pair_len,1)),
    #                                pair[P_X].reshape((pair_len,1)), pair[P_V].reshape((pair_len,1)), np.ones((pair_len,1))*pair[P_L]),axis=1)
    #   Data_arrays = np.concatenate((Data_arrays, np.zeros((1,6)), Data_array)) if i else Data_array
    #   # print(Data_arrays.shape)

    # path = os.path.abspath(os.path.dirname(__file__)) + "\\data_inter\\"
    # np.savetxt(path + 'Data_arrays.csv', Data_arrays, delimiter = ',')

    # for i in range(2):  #len(pairs)
    #     pair = pairs[i]
    #     pair_art = pairs_gen[i]
    #     pair_len = pair[X].size
    #     # print(pair_len)
    #     plt.plot(np.arange(pair_len),
    #              pair[X_VELOCITY],
    #              's-',
    #              color='r',
    #              label='data')
    #     plt.plot(np.arange(pair_len),
    #              pair_art[X_VELOCITY],
    #              'o-',
    #              color='b',
    #              label='gen')
    #     plt.legend(loc='best', fontsize=28)
    #     plt.xlabel('t/dt', fontdict={'family': 'Times New Roman', 'size': 32})
    #     plt.ylabel('Velocity',
    #                fontdict={
    #                    'family': 'Times New Roman',
    #                    'size': 32
    #                })
    #     plt.xticks(size=28)
    #     plt.yticks(size=28)
    #     plt.show()

    # print(len(pairs))

    ###########################################
    ## QP
    ###########################################

    # dynamic
    # h = 0.5950
    # r = 17.5223
    # after
    # h = 0.3869
    # r = 14.9850
    h = 0.3972
    r = 15.0363
    Se = Dhw - h * Vel - r
    dt = 0.04
    A_mat = np.array([[1, dt], [0, 1]])
    B_mat = np.array([-dt * (h + dt), -dt])
    D_mat = np.array([dt, 1])

    # parameters
    d_sigma = 0.406204
    N_data = 50
    beta = 0.2
    # lambda_v = 0.01 * N_data / 719.34  #0.0006 (noise_cov * Bx @ Bx)**(-1)
    lambda_v = 0.01 * N_data / 1096.5  #0.000912
    lambda_c = 2.4
    lambda_b = 0.0005
    random_size = 25
    thres_1 = 40
    thres_2 = 12
    lambda_U_ker = 0.2  # 0.05
    lambda_U_quad = 0.1  # 0.05
    con_v1_ker = 2.5e-1
    con_v2_ker = 2.5e-1
    con_v1_quad = 1e-1
    con_v2_quad = 1e-1

    # data selection
    ind = np.random.randint(1, len(Se), N_data * 2)
    x_cur, x_next, acc, pr_a = [], [], [], []
    x_cur.append(np.array([Se[ind[0]], Nu[ind[0]]]))
    x_next.append(np.array([Se[ind[0] + 1], Nu[ind[0]] + 1]))
    acc.append(np.array([Acc[ind[0]]]))
    pr_a.append(np.array([Pr_a[ind[0]]]))

    for i in range(N_data * 2):
        sign = 0
        for x in x_cur:
            if Se[ind[i]] < x[0] + thres_1 / N_data**0.5 / 5 and Se[
                    ind[i]] > x[0] - thres_1 / 5 / N_data**0.5 and Nu[
                        ind[i]] < x[1] + thres_2 / 5 / N_data**0.5 and Nu[
                            ind[i]] > x[1] - thres_2 / 5 / N_data**0.5:
                sign = 1
                break
        if sign == 0:
            x_cur.append(np.array([Se[ind[i]], Nu[ind[i]]]))
            x_next.append(np.array([Se[ind[i] + 1], Nu[ind[i] + 1]]))
            acc.append(np.array([Acc[ind[i]]]))
            pr_a.append(np.array([Pr_a[ind[i]]]))

        if len(x_cur) == N_data:
            break

    QP_new(x_cur, acc, pr_a, d_sigma, random_size, A_mat, B_mat, D_mat, beta,
           lambda_v, lambda_c, lambda_b, lambda_U_ker, con_v1_ker, con_v2_ker)

    QP_quad(x_cur, acc, pr_a, d_sigma, random_size, A_mat, B_mat, D_mat, beta,
            lambda_v, lambda_c, lambda_b, lambda_U_quad, con_v1_quad,
            con_v2_quad)

    ###########################################
    ## scatter
    ###########################################
    # scatter_plot(Se, Nu)


if __name__ == '__main__':
    created_arguments = create_args()
    print("Try to find the saved pickle file for better performance.")
    # Read the track csv and convert to useful format
    if os.path.exists(created_arguments["pickle_path"]):
        with open(created_arguments["pickle_path"], "rb") as fp:
            tracks = pickle.load(fp)
        print("Found pickle file {}.".format(created_arguments["pickle_path"]))
    else:
        print("Pickle file not found, csv will be imported now.")
        tracks = read_track_csv(created_arguments)
        print("Finished importing the pickle file.")

    if created_arguments["save_as_pickle"] and not os.path.exists(
            created_arguments["pickle_path"]):
        print("Save tracks to pickle file.")
        with open(created_arguments["pickle_path"], "wb") as fp:
            pickle.dump(tracks, fp)

    # Read the static info
    try:
        static_info = read_static_info(created_arguments)
    except:
        print(
            "The static info file is either missing or contains incorrect characters."
        )
        sys.exit(1)

    # Read the video meta
    try:
        meta_dictionary = read_meta_info(created_arguments)
    except:
        print(
            "The video meta file is either missing or contains incorrect characters."
        )
        sys.exit(1)

    # if created_arguments["visualize"]:
    #     if tracks is None:
    #         print("Please specify the path to the tracks csv/pickle file.")
    #         sys.exit(1)
    #     if static_info is None:
    #         print("Please specify the path to the static tracks csv file.")
    #         sys.exit(1)
    #     if meta_dictionary is None:
    #         print("Please specify the path to the video meta csv file.")
    #         sys.exit(1)
    #     visualization_plot = VisualizationPlot(created_arguments, tracks, static_info, meta_dictionary)
    #     visualization_plot.show()

    main(tracks)
