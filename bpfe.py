from bpfe import load


def run():
    for label, data in load.generate_training_rows():
        print label, data
        break


if __name__ == '__main__':
    run()
