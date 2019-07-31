from prepare_data.prepare_data import prepare_input_data


def select_task():
    while True:
        print("Please type a number according to the task you like to run")
        print("1) prepare raw kitti data in model input form -> see prepare_data/prepare_dataset")
        print("2) convert model input data to tfrecords format")
        print("3) play tfrecords files to check data")
        print("4) train sfmlearner model")
        print("5) predict depth and pose from test data")
        print("6) evaluate predicted data")
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
