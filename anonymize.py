import click

from src.anonymization.anonymizer import run_pipeline


@click.command()
@click.option("--in_path", help="Path to input image dir.", type=str)
@click.option("--out_path", help="Path to output image save dir.", type=str)
@click.option("--threshold", help="Detectron2 detection threshold.", type=float, default=0.25)
@click.option("--blur_mode", help="Anonymization blur mode.", type=str, default="median")
@click.option("--detectron_class", help="Detectron2 anonymize class.", type=int, default=0)
def main(in_path, out_path, threshold, blur_mode, detectron_class):
    run_pipeline(
        in_path,
        out_path,
        detectron_threshold=threshold,
        blur_mode=blur_mode,
        detectron_selected_class=detectron_class
    )


if __name__ == "__main__":
    main()
