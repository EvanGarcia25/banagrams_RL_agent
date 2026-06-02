from .episode_visualization import main as server_main


import sys

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    
    if len(argv) == 0:
        argv = [
            "--episode-file",
            "viewer/episodes/demo_seed_7.json",
            "--step-delay",
            "0.7",
        ]
        
    server_main(argv)


if __name__ == "__main__":
    main()
