import yaml


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    cfg = load_config()
    print(cfg)