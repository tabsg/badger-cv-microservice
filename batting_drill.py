from cover_drive_judge import CoverDriveJudge
import sys

def main(input_filename):
    with CoverDriveJudge(input_filename) as judge:
        (averageScore, advice1, advice2) = judge.process_and_write_video()


if __name__ == "__main__":
    main(sys.argv[1])

