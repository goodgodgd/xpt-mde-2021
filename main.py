import numpy as np
from utils.util_funcs import input_integer


def main():
    message = "\nPlease type a number according to the task you like to run \n" \
              "1) prepare raw kitti data in model input form \n" \
              "\t-> see prepare_data/prepare_data_main.py \n" \
              "2) convert model input data to tfrecords format \n" \
              "\t-> see tfrecords/create_tfrecords_main.py \n" \
              "3) train sfmlearner model \n" \
              "\t-> see model/model_main.py \n" \
              "4) predict depth and pose from test data \n" \
              "\t-> see model/model_main.py \n" \
              "5) evaluate predictions \n" \
              "\t-> see evaluate/evaluate_main.py \n" \
              "6) visualize predictions \n" \
              "\t-> see evaluate/visualize_main.py \n"

    task_id = input_integer(message, 1, 6)
    print(f"You selected task #{task_id}")

    if task_id == 1:
        from prepare_data.prepare_data_main import prepare_input_data
        prepare_input_data()
    elif task_id == 2:
        from tfrecords.create_tfrecords_main import convert_to_tfrecords
        convert_to_tfrecords()
    elif task_id == 3:
        from model.model_main import train_by_user_interaction
        train_by_user_interaction()
    elif task_id == 4:
        from model.model_main import predict_by_user_interaction
        predict_by_user_interaction()
    elif task_id == 5:
        from evaluate.evaluate_main import evaluate_by_user_interaction
        evaluate_by_user_interaction()
    elif task_id == 6:
        from evaluate.visualize_main import visualize_by_user_interaction
        visualize_by_user_interaction()


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    main()
