import os
import sys
import pickle
import argparse
import numpy as np

from data_management.read_csv import *
from data_management.functions import *
from visualization.visualize_frame import VisualizationPlot


###########################################
## read and visualize data
###########################################
def create_args():
    parser = argparse.ArgumentParser(description="ParameterOptimizer")
    # --- Input paths ---
    path = "D:/learn/23SS/guided_research/highd-dataset-v1.0/highD-dataset/Python/data/"
    # print(os.getcwd())
    data_num = "01"
    parser.add_argument('--input_path',
                        default=path + data_num + "_tracks.csv",
                        type=str,
                        help='CSV file of the tracks')
    parser.add_argument('--input_static_path',
                        default=path + data_num + "_tracksMeta.csv",
                        type=str,
                        help='Static meta data file for each track')
    parser.add_argument('--input_meta_path',
                        default=path + data_num + "_recordingMeta.csv",
                        type=str,
                        help='Static meta data file for the whole video')
    parser.add_argument(
        '--pickle_path',
        default=path + data_num + ".pickle",
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
    # print('number',len(vf_tracks))

    ###########################################
    ## construct variables needed
    ###########################################
    Vel, Acc, Nu, Dhw, Pr_a, pairs = extract_features(vf_tracks, tracks)

    ###########################################
    ## dynamic least square solver
    ###########################################
    # kp = 0.006354857329713133
    # kd = 0.16748344692513145
    # h = 0.8903920819008573
    kp = 0.017258
    kd = 0.617991
    h = 0.913008
    Se = Dhw - h * Vel  #- r
    dt = 0.04
    A_mat = np.array([[1, dt], [0, 1]])
    B_mat = np.array([-dt * (h + dt), -dt])
    D_mat = np.array([dt, 1])

    ###########################################
    ## calculate and visualize error
    ###########################################
    # Gamma, d_mu, d_sigma = error_cal(pairs,kp,kd,h,A_mat,B_mat,D_mat)
    # print(Gamma, d_mu, d_sigma)
    # with D
    # 719.3441412696383 0.23860922267429896 0.621241544469323
    d_mu, d_sigma = 0, 0.621241544469323

    ###########################################
    ## parameters
    ###########################################
    N_data = 50
    beta = 0.2
    lambda_v = 0.01 * N_data / 719.34  #0.0006
    lambda_c = 1.5
    lambda_b = 0.0005
    random_size = 20
    thres_1 = 40
    thres_2 = 12
    lambda_U = 0.03
    reg_v1_ker = 2.5e-1
    reg_v2_ker = 2.5e-1
    reg_v1_quad = 1e-1
    reg_v2_quad = 1e-1

    ###########################################
    ## QP
    ###########################################
    ind = np.random.randint(1, len(Se), N_data * 2)
    x_cur, x_next, acc, pr_a = [], [], [], []
    x_cur.append(np.array([Se[ind[0]], Nu[ind[0]]]))
    x_next.append(np.array([Se[ind[0] + 1], Nu[ind[0]] + 1]))
    acc.append(np.array([Acc[ind[0]]]))
    # x_last.append(np.array([Se[ind[0] - 1], Nu[ind[0]] - 1]))
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
        if sign == 0 and ind[i] != len(Se) - 1:
            x_cur.append(np.array([Se[ind[i]], Nu[ind[i]]]))
            x_next.append(np.array([Se[ind[i] + 1], Nu[ind[i] + 1]]))
            # x_last.append(np.array([Se[ind[i] - 1], Nu[ind[i] - 1]]))
            acc.append(np.array([Acc[ind[i]]]))
            pr_a.append(np.array([Pr_a[ind[i]]]))

        if len(x_cur) == N_data:
            break

    QP_new(x_cur, x_next, acc, pr_a, d_sigma, random_size, A_mat, B_mat, D_mat,
           beta, lambda_v, lambda_c, lambda_b, lambda_U, reg_v1_ker,
           reg_v2_ker)

    QP_quad(x_cur, x_next, acc, pr_a, d_sigma, random_size, A_mat, B_mat,
            D_mat, beta, lambda_v, lambda_c, lambda_b, lambda_U, reg_v1_quad,
            reg_v2_quad)
    ###########################################
    ## scatter
    ###########################################
    # scatter_plot(Se, Nu)


if __name__ == '__main__':
    created_arguments = create_args()
    # print("Try to find the saved pickle file for better performance.")
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
