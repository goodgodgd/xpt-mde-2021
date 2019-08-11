from prepare_data.prepare_data_main import prepare_input_data


def select_task():
    while True:
        print("Please type a number according to the task you like to run")
        print("1) prepare raw kitti data in model input form -> see prepare_data/prepare_data_main.py")
        print("2) convert model input data to tfrecords format -> see tfrecords/create_tfrecords_main.py")
        print("3) train sfmlearner model -> see ")
        print("4) predict depth and pose from test data")
        print("5) evaluate predicted data")
        key = input()
        print(key)
        try:
            key = int(key)
            break
        except Exception as e:
            print("Please type only a NUMBER")

    print(f"You selected task #{key}")
    return key


def run_task(task_id):
    if task_id == 1:
        prepare_input_data()


def main():
    task_id = select_task()
    run_task(task_id)


if __name__ == "__main__":
    main()
