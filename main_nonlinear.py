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
    V_gen, A_gen, Nu_gen, D_gen, Pr_x_a_gen, pairs_gen = extract_features_gen(
        vf_tracks, tracks)
    # print(len(Vel))

    ###########################################
    ## construct data_array exported to MATLAB
    ###########################################
    P_V = "p_v"
    P_X = "p_x"
    P_L = 'p_l'
    X = "x"

    # for i in range(len(pairs)):
    #   pair = pairs[i]
    #   pair_len = pair[X].size
    #   Data_array = np.concatenate((pair[X].reshape((pair_len,1)), pair[X_VELOCITY].reshape((pair_len,1)), pair[X_ACCELERATION].reshape((pair_len,1)),
    #                                pair[P_X].reshape((pair_len,1)), pair[P_V].reshape((pair_len,1)), np.ones((pair_len,1))*pair[P_L]),axis=1)
    #   Data_arrays = np.concatenate((Data_arrays, np.zeros((1,6)), Data_array)) if i else Data_array
    #   # print(Data_arrays.shape)

    # path = os.path.abspath(os.path.dirname(__file__)) + "\\data_inter\\"
    # np.savetxt(path + 'Data_arrays.csv', Data_arrays, delimiter = ',')

    for i in range(2):  #len(pairs)
        pair = pairs[i]
        pair_art = pairs_gen[i]
        pair_len = pair[X].size
        # print(pair_len)
        plt.plot(np.arange(pair_len),
                 pair[X_VELOCITY],
                 's-',
                 color='r',
                 label='data')
        plt.plot(np.arange(pair_len),
                 pair_art[X_VELOCITY],
                 'o-',
                 color='b',
                 label='gen')
        plt.legend(loc='best', fontsize=28)
        plt.xlabel('t/dt', fontdict={'family': 'Times New Roman', 'size': 32})
        plt.ylabel('Velocity',
                   fontdict={
                       'family': 'Times New Roman',
                       'size': 32
                   })
        plt.xticks(size=28)
        plt.yticks(size=28)
        plt.show()

    # print(len(pairs))


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
