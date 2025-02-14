from bytelatent.args import TrainArgs
from bytelatent.config_parser import parse_args_to_pydantic_model


def main():
    train_args = parse_args_to_pydantic_model(TrainArgs)
    print(train_args.model_dump_json(indent=4))


if __name__ == "__main__":
    main()
