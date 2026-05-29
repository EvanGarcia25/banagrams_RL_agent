from .episode_visualization import main as server_main


def main(argv=None):
    server_main([
        "--episode-file",
        "viewer/episodes/demo_seed_7.json",
        "--step-delay",
        "0.7",
    ])


if __name__ == "__main__":
    main()
