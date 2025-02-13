from bytelatent.args import TrainArgs, parse_args


def main():
    train_args = parse_args(TrainArgs)
    print(train_args.model_dump_json(indent=4))


if __name__ == "__main__":
    main()
