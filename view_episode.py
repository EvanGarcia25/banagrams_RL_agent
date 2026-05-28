from server import main as server_main


if __name__ == "__main__":
    server_main([
        "--episode-file",
        "episodes/demo_seed_7.json",
        "--step-delay",
        "0.7",
    ])