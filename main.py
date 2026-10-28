"""Run the full pipeline: clean raw data -> exploratory analysis -> models."""

import analysis
import data_cleaning
import model


def main() -> None:
    df = data_cleaning.clean()
    analysis.run(df)
    model.run(df)


if __name__ == "__main__":
    main()
